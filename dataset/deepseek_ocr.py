
import os
import logging
from typing import List, Optional, Union
from pathlib import Path
import torch
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

class DeepSeekOCRProcessor:
    """
    Wrapper for DeepSeek-OCR (DeepSeek-VL) to handle PDF and Image-to-Text conversion.
    
    This class abstracts the complexity of loading the model and processing different
    file formats. It includes graceful fallback if dependencies are missing.
    """
    
    def __init__(self, model_path: str = "deepseek-ai/deepseek-vl-1.3b-chat", device: str = "cuda"):
        """
        Initialize the DeepSeek-OCR processor.
        
        Args:
            model_path: HuggingFace model path or local path.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_available = False
        
        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load DeepSeek-VL model and processor."""
        try:
            from transformers import AutoModelForCausalLM
            from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
            from deepseek_vl.utils.io import load_pil_images
            
            logger.info(f"Loading DeepSeek-VL model from {self.model_path} on {self.device}...")
            
            self.processor = VLChatProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.processor.tokenizer
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            self._is_available = True
            logger.info("DeepSeek-OCR loaded successfully.")
            
        except ImportError:
            logger.warning("DeepSeek-VL dependencies not found. OCR will run in fallback mode (placeholder).")
            logger.warning("Please install: pip install deepseek-vl")
        except Exception as e:
            logger.error(f"Failed to load DeepSeek-VL: {e}")
            self._is_available = False

    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self._is_available

    def process_image(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from a single image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Extracted text.
        """
        if not self._is_available:
            return f"[OCR UNAVAILABLE] Could not process {image_path}"
            
        try:
            from deepseek_vl.utils.io import load_pil_images
            
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>Describe this image in detail and transcribe any text visible.",
                    "images": [str(image_path)]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            
            # Prepare inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True
                )
                
            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            return answer
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return f"[OCR ERROR] {str(e)}"

    def process_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from a PDF by converting pages to images.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Combined extracted text from all pages.
        """
        if not self._is_available:
            return f"[OCR UNAVAILABLE] Could not process {pdf_path}"
            
        try:
            import pypdf
            from pdf2image import convert_from_path
            
            # First try simple text extraction with pypdf
            text_content = []
            try:
                reader = pypdf.PdfReader(pdf_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
            except Exception:
                pass
                
            # If pypdf failed or returned empty, use visual OCR
            if not text_content:
                logger.info(f"PDF text extraction empty, switching to visual OCR for {pdf_path}")
                images = convert_from_path(str(pdf_path))
                
                for i, image in enumerate(images):
                    # Save temp image
                    temp_path = f"temp_page_{i}.jpg"
                    image.save(temp_path, "JPEG")
                    
                    # Process with DeepSeek
                    page_text = self.process_image(temp_path)
                    text_content.append(f"--- Page {i+1} ---\n{page_text}")
                    
                    # Cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            return "\n\n".join(text_content)
            
        except ImportError:
            return "[OCR ERROR] Missing PDF dependencies. Install pypdf and pdf2image."
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return f"[OCR ERROR] {str(e)}"
