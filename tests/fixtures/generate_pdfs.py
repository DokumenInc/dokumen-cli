"""Generate PDF fixtures for testing."""
import sys
from pathlib import Path

# Add parent directory to path to import conftest
sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import create_minimal_pdf, create_large_pdf

# Create fixtures directory
fixtures_dir = Path(__file__).parent

# Create sample.pdf
sample_pdf = fixtures_dir / "sample.pdf"
sample_pdf.write_bytes(create_minimal_pdf())
print(f"Created: {sample_pdf} ({sample_pdf.stat().st_size} bytes)")

# Create large.pdf (4.5 MB)
large_pdf = fixtures_dir / "large.pdf"
large_pdf.write_bytes(create_large_pdf(4.5))
print(f"Created: {large_pdf} ({large_pdf.stat().st_size / 1024 / 1024:.2f} MB)")

print("PDF fixtures generated successfully!")
