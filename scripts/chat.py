"""
Ollama-style Terminal Chat Interface for TRM Model

Based on research from:
- https://github.com/jeromeboivin/ollama-chat
- PyTorch inference best practices
"""

import torch
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
import time
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_model import ModelLoader
from dataset.common import get_tokenizer


class ChatSession:
    """Manage chat conversation state and history."""
    
    def __init__(self, model, tokenizer, config: Dict, system_prompt: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.conversation_history: List[Dict] = []
        self.max_seq_len = config.get('seq_len', 511)
        self.temperature = 0.8
        
    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def format_prompt(self, user_input: str) -> str:
        """Format conversation history into prompt."""
        # Simple format: System + conversation history
        prompt_parts = [f"System: {self.system_prompt}\n"]
        
        # Add conversation history (last N turns to fit context)
        for msg in self.conversation_history[-5:]:  # Keep last 5 exchanges
            role = msg['role'].capitalize()
            content = msg['content']
            prompt_parts.append(f"{role}: {content}\n")
        
        # Add current user input
        prompt_parts.append(f"User: {user_input}\n")
        prompt_parts.append("Assistant:")
        
        return "".join(prompt_parts)
    
    def generate_response(self, user_input: str, max_new_tokens: int = 256, 
                         verbose: bool = False) -> str:
        """
        Generate model response using greedy decoding.
        
        Args:
            user_input: User's input text
            max_new_tokens: Maximum tokens to generate
            verbose: Print generation stats
            
        Returns:
            generated_text: Model's response
        """
        # Format prompt
        prompt = self.format_prompt(user_input)
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        # Pad to max_seq_len to avoid dimension mismatches during generation
        # (model's hidden states are sized to sequence length)
        if len(input_ids) < self.max_seq_len:
            # Pad with pad_token_id at the beginning
            padding_length = self.max_seq_len - len(input_ids)
            input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
        elif len(input_ids) > self.max_seq_len:
            # Truncate if too long
            input_ids = input_ids[-self.max_seq_len:]
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.model.inner.embed_tokens.embedding_weight.device)
        
        # Prepare batch
        batch = {
            'inputs': input_tensor,
            'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=input_tensor.device),
            'labels': torch.full_like(input_tensor, -100)
        }
        
        # Generate
        start_time = time.time()
        generated_ids = []
        
        with torch.no_grad():
            # Initialize carry
            carry = self.model.initial_carry(batch)
            
            for step in range(max_new_tokens):
                # Forward pass
                carry, outputs = self.model(carry, batch)
                
                # Handle dict or tensor outputs
                if isinstance(outputs, dict):
                    # Deep supervision or multi-output mode
                    if 'final' in outputs:
                        logits = outputs['final'][0, -1, :]  # [vocab_size]
                    elif 'logits' in outputs:
                        logits = outputs['logits'][0, -1, :]
                    else:
                        # Fallback: use first value
                        logits = list(outputs.values())[0][0, -1, :]
                else:
                    # Direct tensor output
                    logits = outputs[0, -1, :]  # [vocab_size]
                
                # Temperature sampling
                if self.temperature > 0:
                    probs = torch.softmax(logits / self.temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    # Greedy
                    next_token = logits.argmax().item()
                
                generated_ids.append(next_token)
                
                # Debug first few tokens
                if verbose and step < 5:
                    token_text = self.tokenizer.decode([next_token])
                    print(f"[Step {step}: token={next_token}, text='{token_text}']")
                
                # Check for EOS token
                if next_token == self.tokenizer.eos_token_id:
                    if verbose:
                        print(f"[Hit EOS at step {step}]")
                    break
                
                # Update batch for next iteration
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=input_tensor.device)
                batch['inputs'] = torch.cat([batch['inputs'], next_token_tensor], dim=1)
                
                # Truncate if exceeds max length
                if batch['inputs'].shape[1] > self.max_seq_len:
                    batch['inputs'] = batch['inputs'][:, -self.max_seq_len:]
                
                batch['labels'] = torch.full_like(batch['inputs'], -100)
                
                # CRITICAL: Reinitialize carry when sequence length changes
                # The carry contains tensors sized for the sequence length
                carry = self.model.initial_carry(batch)
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        elapsed = time.time() - start_time
        tokens_per_sec = len(generated_ids) / elapsed if elapsed > 0 else 0
        
        if verbose:
            print(f"\n[Generated {len(generated_ids)} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} t/s)]")
        
        return generated_text.strip()
    
    def save_conversation(self, filepath: str):
        """Save conversation history to JSON."""
        data = {
            'system_prompt': self.system_prompt,
            'conversation': self.conversation_history,
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'hidden_size': self.config.get('hidden_size'),
                'vocab_size': self.config.get('vocab_size'),
                'H_cycles': self.config.get('H_cycles'),
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Conversation saved to: {filepath}")
    
    def load_conversation(self, filepath: str):
        """Load conversation history from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.system_prompt = data.get('system_prompt', self.system_prompt)
        self.conversation_history = data.get('conversation', [])
        
        print(f"📂 Loaded {len(self.conversation_history)} messages from: {filepath}")


class OllamaStyleChat:
    """Ollama-style terminal chat interface."""
    
    def __init__(self, model_path: str, device: str = 'cpu', 
                 system_prompt: Optional[str] = None, temperature: float = 0.8,
                 tokenizer_name: str = 'gpt2'):
        """
        Initialize chat interface.
        
        Args:
            model_path: Path to model checkpoint
            device: 'cpu' or 'cuda'
            system_prompt: System prompt for the model
            temperature: Sampling temperature (0 = greedy)
            tokenizer_name: Tokenizer to use
        """
        print("🔄 Loading model...")
        
        # Load model
        loader = ModelLoader(model_path, device)
        self.model = loader.load_model(quantized='bnn' in model_path.lower() or 'quantized' in model_path.lower())
        self.config = loader.config
        
        # Load tokenizer
        print(f"🔄 Loading tokenizer: {tokenizer_name}")
        self.tokenizer = get_tokenizer(tokenizer_name)
        
        # Create session
        self.session = ChatSession(self.model, self.tokenizer, self.config, system_prompt)
        self.session.temperature = temperature
        
        print("\n✅ Chat interface ready!")
    
    def run_interactive(self, auto_save: bool = False, conversations_folder: str = "."):
        """Run interactive chat loop (Ollama-style)."""
        print("\n" + "="*70)
        print("  💬 TRM Chat - Ollama Style Terminal Interface")
        print("="*70)
        print(f"\nSystem prompt: {self.session.system_prompt}")
        print(f"Temperature: {self.session.temperature}")
        print(f"Model: {self.config.get('name', 'TRM')}")
        print(f"Device: {self.model.inner.embed_tokens.embedding_weight.device}")
        print("\nCommands:")
        print("  /help     - Show this help")
        print("  /clear    - Clear conversation history")
        print("  /save     - Save conversation")
        print("  /load     - Load conversation")
        print("  /temp X   - Set temperature to X")
        print("  /exit     - Exit chat")
        print("\n" + "="*70 + "\n")
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    cmd = user_input.split()[0].lower()
                    
                    if cmd == '/exit' or cmd == '/quit':
                        if auto_save:
                            filepath = Path(conversations_folder) / f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            self.session.save_conversation(str(filepath))
                        print("\n👋 Goodbye!")
                        break
                    
                    elif cmd == '/help':
                        print("\nAvailable commands:")
                        print("  /help     - Show this help")
                        print("  /clear    - Clear conversation history")
                        print("  /save     - Save conversation")
                        print("  /load     - Load conversation")
                        print("  /temp X   - Set temperature to X")
                        print("  /exit     - Exit chat\n")
                        continue
                    
                    elif cmd == '/clear':
                        self.session.conversation_history = []
                        print("🗑️  Conversation cleared.\n")
                        continue
                    
                    elif cmd == '/save':
                        filepath = Path(conversations_folder) / f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        self.session.save_conversation(str(filepath))
                        continue
                    
                    elif cmd == '/load':
                        filepath = input("Enter filepath: ").strip()
                        try:
                            self.session.load_conversation(filepath)
                        except Exception as e:
                            print(f"❌ Error loading: {e}")
                        continue
                    
                    elif cmd == '/temp':
                        try:
                            temp = float(user_input.split()[1])
                            self.session.temperature = temp
                            print(f"🌡️  Temperature set to: {temp}\n")
                        except (IndexError, ValueError):
                            print("❌ Usage: /temp <value>\n")
                        continue
                    
                    else:
                        print(f"❌ Unknown command: {cmd}\n")
                        continue
                
                # Generate response
                print("Assistant: ", end='', flush=True)
                
                try:
                    response = self.session.generate_response(user_input, verbose=True)
                    print(response)
                    
                    # Add to history
                    self.session.add_message('user', user_input)
                    self.session.add_message('assistant', response)
                    
                except Exception as e:
                    import traceback
                    print(f"\n❌ Error generating response: {e}")
                    print("\nFull traceback:")
                    traceback.print_exc()
                
                print()  # Empty line for readability
        
        except KeyboardInterrupt:
            if auto_save:
                filepath = Path(conversations_folder) / f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.session.save_conversation(str(filepath))
            print("\n\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description='Ollama-style terminal chat interface for TRM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic chat with BNN model
  python chat.py --model model_cpu_bnn.pt
  
  # Chat with custom system prompt
  python chat.py --model checkpoints/text-trm-10m/latest.pt --system-prompt "You are a helpful coding assistant."
  
  # Chat with auto-save
  python chat.py --model model_cpu_bnn.pt --auto-save --conversations-folder ./chats
  
  # Chat with higher temperature (more creative)
  python chat.py --model model_cpu_bnn.pt --temperature 1.0
        """
    )
    
    parser.add_argument('--model', '--checkpoint', type=str, required=True, dest='model',
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run on (default: cpu)')
    parser.add_argument('--system-prompt', type=str, default=None,
                       help='System prompt for the model')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0=greedy, 1.0=creative, default: 0.8)')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer name (default: gpt2)')
    parser.add_argument('--auto-save', action='store_true',
                       help='Auto-save conversation on exit')
    parser.add_argument('--conversations-folder', type=str, default='.',
                       help='Folder to save conversations (default: current dir)')
    
    args = parser.parse_args()
    
    # Create conversations folder if needed
    if args.auto_save:
        Path(args.conversations_folder).mkdir(parents=True, exist_ok=True)
    
    # Launch chat
    chat = OllamaStyleChat(
        model_path=args.model,
        device=args.device,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        tokenizer_name=args.tokenizer
    )
    
    chat.run_interactive(
        auto_save=args.auto_save,
        conversations_folder=args.conversations_folder
    )


if __name__ == "__main__":
    main()
