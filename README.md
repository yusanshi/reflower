# Reflower

Reflow a PDF file for e-readers like Kindle.

- Support English documents only.
- Support one-column or two-columns documents.
- Designed for academic papers (see the example below).

## Get Started

### Requirements
- Linux-based OS
- Python 3.6+

### Preparations
- Install tesseract-related packages: install [tesseract-ocr](https://github.com/tesseract-ocr/tesseract) (I use Tesseract 5 and [tessdata_best models](https://github.com/tesseract-ocr/tessdata_best/blob/main/eng.traineddata)) and pytesseract (`pip install pytesseract`)
- Install layout-parser: `pip install "layoutparser[layoutmodels]"`
- Install OpenCV-Python
- Other packages: `pip install Pillow pdf2image numpy`

### Run

```bash
# For a single file
python reflower.py --source ./input.pdf --target ./output.pdf --target_paper pw3

# Parallel processing for multiple files
sudo apt install parallel
mkdir -p output log
find input/ -name "*.pdf" | parallel -j 4 --bar --results log python reflower.py --source {} --target ./output/{/} --target_paper pw3
find log -type f -name stderr -not -empty -printf '\n==> %p <==\n' -exec cat {} \;
```

## Example

Click the filename to download the PDF file, click the image to view in a new tab.

[**input.pdf**](https://github.com/yusanshi/reflower/files/7653823/input.pdf)

![input](https://user-images.githubusercontent.com/36265606/144707573-cad35f68-1568-42a8-9136-76500164e6da.jpg)

[**intermediate.pdf**](https://github.com/yusanshi/reflower/files/7653824/intermediate.pdf)

![intermediate](https://user-images.githubusercontent.com/36265606/144707579-7da7184f-465f-4f5a-8fe9-239a871e942b.jpg)

[**output.pdf**](https://github.com/yusanshi/reflower/files/7653825/output.pdf)

![output](https://user-images.githubusercontent.com/36265606/144707583-c349ee33-4519-4c87-807f-4a7000642326.jpg)

## TODO

- Copying vector text instead of rasterized text (need to first convert text in pdf to outlines). But this may slow down a PDF reader so will not be suitable for e-readers like Kindle?
- Support scaling
- Don't do a second OCR with ocrmypdf, instead use the first OCR results to create the invisible text layer. (Update: ocrmypdf has been removed, currently no text layer is added. If you need this, simply use ocrmypdf cli)
- Bad results with a document with many inline formulas (mainly because of poor OCR results)
- Too slow :(
