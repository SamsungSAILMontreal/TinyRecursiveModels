"""
Text Rendering for Vision-Unified Architecture

Converts text → images so all inputs can be processed through vision encoder.
Supports plain text, code (syntax-highlighted), and structured text.
"""

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from typing import Union, List, Optional, Tuple
import textwrap
import subprocess
import os


def ensure_dejavu_font_installed():
    """Install DejaVuSansMono font on Colab/Linux if not available."""
    font_path = "/usr/local/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    
    # Check if already installed and working
    try:
        font = ImageFont.truetype(font_path, 14)
        return font_path
    except:
        pass
    
    print("Installing DejaVuSansMono font...")
    
    if os.name == 'nt':
        # print("Windows detected: Skipping font installation. Please install DejaVuSansMono manually if needed.")
        return "arial.ttf"  # Fallback for Windows

    try:
        # Clean up any broken files
        subprocess.run(["rm", "-f", font_path], check=False, stderr=subprocess.DEVNULL)
        
        # Create directory
        os.makedirs("/usr/local/share/fonts/truetype/dejavu", exist_ok=True)
        
        # Download official .deb package
        subprocess.run([
            "wget", "-q",
            "https://ftp.debian.org/debian/pool/main/f/fonts-dejavu/fonts-dejavu-core_2.37-2_all.deb",
            "-O", "/tmp/fonts-dejavu.deb"
        ], check=True, stderr=subprocess.DEVNULL)
        
        # Extract TTF file
        subprocess.run(["dpkg-deb", "-x", "/tmp/fonts-dejavu.deb", "/tmp/dejavu_fonts"],
                      check=True, stderr=subprocess.DEVNULL)
        
        # Copy to system fonts
        subprocess.run([
            "bash", "-c",
            r"find /tmp/dejavu_fonts -name 'DejaVuSansMono.ttf' -exec cp {} /usr/local/share/fonts/truetype/dejavu/ \;"
        ], check=True, stderr=subprocess.DEVNULL)
        
        # Refresh font cache
        subprocess.run(["fc-cache", "-fv"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Verify installation
        font = ImageFont.truetype(font_path, 14)
        print("DejaVuSansMono font installed successfully!")
        return font_path
        
    except Exception as e:
        print(f"Font installation failed: {e}")
        return None


class TextRenderer:
    """
    Render text as images for vision encoder input.
    
    Supports:
    - Plain text with word wrapping
    - Syntax-highlighted code (via pygments)
    - Configurable fonts, sizes, colors
    - Variable resolution output
    """
    
    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        font_size: int = 12,
        font_family: str = "DejaVuSansMono.ttf",  # Monospace for code
        bg_color: str = "white",
        text_color: str = "black",
        padding: int = 10,
    ):
        """
        Args:
            width: Output image width in pixels
            height: Output image height in pixels
            font_size: Font size in points
            font_family: Font family name or path to .ttf file
            bg_color: Background color (name or hex)
            text_color: Text color (name or hex)
            padding: Padding around text in pixels
        """
        self.width = width
        self.height = height
        self.font_size = font_size
        self.bg_color = bg_color
        self.text_color = text_color
        self.padding = padding
        
        # Try to load font, fallback to default if not found
        font_loaded = False
        
        # Try system font first
        try:
            self.font = ImageFont.truetype(font_family, font_size)
            font_loaded = True
        except IOError:
            pass
        
        if not font_loaded:
            # Try common paths
            font_paths = [
                f"/usr/local/share/fonts/truetype/dejavu/{font_family}",
                f"/usr/share/fonts/truetype/dejavu/{font_family}",
                f"C:\\Windows\\Fonts\\{font_family}",
                f"/System/Library/Fonts/{font_family}",
            ]
            for path in font_paths:
                if os.path.exists(path):
                    try:
                        self.font = ImageFont.truetype(path, font_size)
                        font_loaded = True
                        break
                    except:
                        continue
        
        # If still not found, try automatic installation (Linux/Colab)
        if not font_loaded and font_family == "DejaVuSansMono.ttf":
            installed_path = ensure_dejavu_font_installed()
            if installed_path:
                try:
                    self.font = ImageFont.truetype(installed_path, font_size)
                    font_loaded = True
                except:
                    pass
        
        # Final fallback to default PIL font
        if not font_loaded:
            print(f"Font {font_family} not found, using default")
            self.font = ImageFont.load_default()
        
        # Calculate usable text area
        self.text_width = width - 2 * padding
        self.text_height = height - 2 * padding
        
        # Estimate characters per line (rough approximation)
        try:
            # Get average character width
            test_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            bbox = self.font.getbbox(test_str)
            avg_char_width = (bbox[2] - bbox[0]) / len(test_str)
            self.chars_per_line = int(self.text_width / avg_char_width)
        except:
            # Fallback estimate
            self.chars_per_line = max(40, self.text_width // 8)
    
    def render_plain_text(self, text: str) -> Image.Image:
        """
        Render plain text with word wrapping.
        
        Args:
            text: Text string to render
            
        Returns:
            PIL Image
        """
        # Create blank image
        img = Image.new('RGB', (self.width, self.height), color=self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Word wrap text
        lines = []
        for paragraph in text.split('\n'):
            if paragraph.strip():
                wrapped = textwrap.fill(
                    paragraph, 
                    width=self.chars_per_line,
                    break_long_words=True,
                    break_on_hyphens=False
                )
                lines.extend(wrapped.split('\n'))
            else:
                lines.append('')  # Preserve empty lines
        
        # Draw text line by line
        y = self.padding
        line_height = self.font_size + 4  # Add spacing between lines
        
        for line in lines:
            if y + line_height > self.height - self.padding:
                # Truncate if text exceeds image height
                break
            
            draw.text((self.padding, y), line, fill=self.text_color, font=self.font)
            y += line_height
        
        return img
    
    def render_code(
        self, 
        code: str, 
        language: Optional[str] = None,
        syntax_highlight: bool = True
    ) -> Image.Image:
        """
        Render code with optional syntax highlighting.
        
        Args:
            code: Code string to render
            language: Programming language (auto-detect if None)
            syntax_highlight: Enable syntax highlighting (requires pygments)
            
        Returns:
            PIL Image
        """
        if syntax_highlight:
            try:
                from pygments import highlight
                from pygments.lexers import get_lexer_by_name, guess_lexer
                from pygments.formatters import ImageFormatter
                from pygments.styles import get_style_by_name
                
                # Determine lexer
                if language:
                    lexer = get_lexer_by_name(language)
                else:
                    lexer = guess_lexer(code)
                
                # Create formatter with our settings
                formatter = ImageFormatter(
                    font_name='DejaVu Sans Mono',
                    font_size=self.font_size,
                    line_numbers=False,
                    style='default',  # or 'monokai', 'github', etc.
                    image_format='PNG'
                )
                
                # Render to bytes
                output = io.BytesIO()
                highlight(code, lexer, formatter, output)
                output.seek(0)
                
                # Load as PIL image and resize to target dimensions
                img = Image.open(output)
                img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
                
                return img
                
            except ImportError:
                print("WARNING: pygments not installed, rendering as plain text")
                syntax_highlight = False
            except Exception as e:
                # Suppress repetitive font warnings (already shown during init)
                if not hasattr(self, '_font_warning_shown'):
                    pass  # Warning already shown in __init__
                syntax_highlight = False
        
        # Fallback to plain text rendering
        return self.render_plain_text(code)
    
    def render_batch(
        self, 
        texts: List[str],
        detect_code: bool = True,
        language: Optional[str] = None
    ) -> np.ndarray:
        """
        Render multiple texts as batch of images.
        
        Args:
            texts: List of text strings
            detect_code: Auto-detect and highlight code
            language: Force language for syntax highlighting
            
        Returns:
            numpy array [batch, height, width, 3] in RGB format
        """
        images = []
        
        for text in texts:
            # Simple heuristic: detect code by presence of common markers
            is_code = detect_code and self._looks_like_code(text)
            
            if is_code:
                img = self.render_code(text, language=language)
            else:
                img = self.render_plain_text(text)
            
            # Convert to numpy array
            img_array = np.array(img)
            images.append(img_array)
        
        # Stack into batch
        return np.stack(images, axis=0)
    
    def _looks_like_code(self, text: str) -> bool:
        """
        Heuristic to detect if text looks like code.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text appears to be code
        """
        code_markers = [
            'def ', 'class ', 'import ', 'from ',  # Python
            'function ', 'const ', 'let ', 'var ',  # JavaScript
            'public ', 'private ', 'void ', 'int ',  # Java/C++
            '{', '}', ';', '()', '[]',  # Common syntax
            '==', '!=', '+=', '->',  # Operators
        ]
        
        # Count code markers
        marker_count = sum(1 for marker in code_markers if marker in text)
        
        # If multiple markers present, likely code
        return marker_count >= 2
    
    def render_with_title(
        self, 
        text: str, 
        title: str,
        title_bg: str = "lightgray"
    ) -> Image.Image:
        """
        Render text with a title bar (useful for documents/sections).
        
        Args:
            text: Main text content
            title: Title text
            title_bg: Background color for title bar
            
        Returns:
            PIL Image with title bar
        """
        # Create image
        img = Image.new('RGB', (self.width, self.height), color=self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw title bar
        title_height = self.font_size + 8
        draw.rectangle(
            [(0, 0), (self.width, title_height)],
            fill=title_bg
        )
        draw.text(
            (self.padding, 4),
            title,
            fill=self.text_color,
            font=self.font
        )
        
        # Draw main text below title
        y = title_height + self.padding
        lines = textwrap.fill(text, width=self.chars_per_line).split('\n')
        line_height = self.font_size + 4
        
        for line in lines:
            if y + line_height > self.height - self.padding:
                break
            draw.text((self.padding, y), line, fill=self.text_color, font=self.font)
            y += line_height
        
        return img


def render_text_to_image(
    text: Union[str, List[str]],
    width: int = 224,
    height: int = 224,
    font_size: int = 12,
    detect_code: bool = True,
    language: Optional[str] = None
) -> Union[Image.Image, np.ndarray]:
    """
    Convenience function to render text as image(s).
    
    Args:
        text: Single text string or list of strings
        width: Image width
        height: Image height
        font_size: Font size in points
        detect_code: Auto-detect and syntax-highlight code
        language: Force programming language for highlighting
        
    Returns:
        Single PIL Image if text is string, numpy array if list
    """
    renderer = TextRenderer(width=width, height=height, font_size=font_size)
    
    if isinstance(text, str):
        # Single text
        if detect_code and renderer._looks_like_code(text):
            return renderer.render_code(text, language=language)
        else:
            return renderer.render_plain_text(text)
    else:
        # Batch of texts
        return renderer.render_batch(text, detect_code=detect_code, language=language)


if __name__ == "__main__":
    # Test rendering
    print("Testing TextRenderer...")
    
    # Test 1: Plain text
    sample_text = "This is a test of text rendering for vision models. The quick brown fox jumps over the lazy dog."
    renderer = TextRenderer(width=224, height=224, font_size=14)
    img = renderer.render_plain_text(sample_text)
    img.save("test_plain.png")
    print("[OK] Saved test_plain.png")
    
    # Test 2: Code
    sample_code = """def hello_world():
    print("Hello, World!")
    return 42
"""
    img = renderer.render_code(sample_code, language='python')
    img.save("test_code.png")
    print("[OK] Saved test_code.png")
    
    # Test 3: Batch rendering
    texts = [
        "First text sample",
        "def test(): pass",
        "Another plain text"
    ]
    batch = renderer.render_batch(texts)
    print(f"[OK] Batch shape: {batch.shape}")
    
    print("\nAll tests passed!")
