# Reflower

```bash
python reflower.py --source input.pdf --target ./output.pdf --target_paper pw3

# or

sudo apt install parallel
mkdir -p output log
find input/ -name "*.pdf" | parallel -j 4 --bar --results log python reflower.py --source {} --target ./output/{/} --dpi 300 --target_paper pw3
find log -type f -name stderr -not -empty -printf '\n==> %p <==\n' -exec cat {} \;
```

## Example

**Input** [input.pdf](https://storage.yusanshi.com/reflower/input.pdf)

print with multiple sheets in one page and convert it to png...

**Intermediate** [intermediate.pdf](https://storage.yusanshi.com/reflower/intermediate.pdf)

**Output** [output.pdf](https://storage.yusanshi.com/reflower/output.pdf)


## TODO
- add invisible text layer
