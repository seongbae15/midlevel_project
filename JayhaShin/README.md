"midlevel_depth_MIDAS.py"
 - Input: image
 - Output: depthmap
 - Apple M1 Chip에서 모델 [MIDAS] 구동시 다음 에러 발생.
    (Intel MKL FATAL ERROR: This system does not meet the minimum requirements
    for use of the Intel(R) Math Kernel Library.)
 - 빠른 해결 방안: Google Colab
<br /><br />

"midlevel_price_OCR_easyocr.py"
 - Input: image (매대, 가격표 등)
 - Output: Radial heatmap
 - 인식률 개선 필요
<br /><br />

"midlevel_price_OCR_pytesseract.py"
 - Input: image (매대, 가격표 등)
 - Output: Radial heatmap
 - 인식률 개선 필요
<br /><br />

"_ocr_common_func.py"
 - OCR 작업 관련 공통 함수 모음