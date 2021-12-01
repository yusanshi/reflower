# Reflower

```bash
python reflower.py --source input.pdf --target ./output.pdf --dpi 300 --target_paper pw3

# or

sudo apt install parallel
mkdir -p output log
find input/ -name "*.pdf" | parallel -j 4 --bar --results log python reflower.py --source {} --target ./output/{/} --dpi 300 --target_paper pw3
find log -type f -name stderr -not -empty -printf '\n==> %p <==\n' -exec cat {} \;
```
## TODO
- add invisible text layer
