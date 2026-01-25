from app.reporting import render_pdf

def test_pdf_generation():
    print("Testing PDF Generation...")
    sample_md = """
# Test Report
## Section 1
This is a test of the **PDF generation** system.

| Col A | Col B |
|-------|-------|
| Val 1 | Val 2 |
    """
    
    try:
        pdf_bytes = render_pdf(sample_md)
        # Check PDF signature
        if pdf_bytes.startswith(b'%PDF'):
            print("SUCCESS: Generated valid PDF bytes.")
            print(f"PDF Size: {len(pdf_bytes)} bytes")
        else:
            print("FAILURE: Output does not look like a PDF.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        # Print traceback to debug library issues
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_generation()
