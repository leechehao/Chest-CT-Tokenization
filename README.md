報告切完 Event 之後，將 COMP、FIND 及 IMP 以 `,`、`(`、`)`、`[`、`]`、`:` 的前後都加空白。
```python
clean_report = re.sub(r"([\[()\]:,])", r" \1 ", report).strip()
clean_report = re.sub(r"\s+", " ", clean_report)
```