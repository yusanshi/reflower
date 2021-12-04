# Reflower

Reflow a PDF file for e-readers like Kindle.

## Get Started

### Preparations

TODO
install tesseract and use the tesseract_best
pip install ...

### Run

```bash
python reflower.py --source ./input.pdf --target ./output.pdf --target_paper pw3

# or

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

