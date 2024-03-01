# Chest CT Tokenization
針對 Chest CT 影像文字報告，訓練模型解決連字問題。

## 文本前處理
先將文本中的 `,`、`(`、`)`、`[`、`]`、`:` 前後都加上空白，再經由模型處理。
```python
text = "1. A 2cm massin the rightupperlobe(RUL), highly suspicious for primary lungcancer."
clean_text = re.sub(r"([\[()\]:,])", r" \1 ", text).strip()
clean_text = re.sub(r"\s+", " ", clean_text)
print(clean_text) # 1. A 2cm massin the rightupperlobe ( RUL ) , highly suspicious for primary lungcancer.
```

## 訓練模型
訓練模型的程式碼是使用自己開發的 [winlp](https://github.com/leechehao/MyMLOps) 套件，它主要結合了 PyTorch Lightning 和 Hydra 的強大功能。

## Inference
```bash
python inference.py --tracking_uri http://192.168.1.76:9487 \
                    --run_id 7d4f5433a4a848ad9b68560848988db5 \
                    --text "1. A 2cm massin the rightupperlobe(RUL), highly suspicious for primary lungcancer."
```
+ **tracking_uri** *(str)* ─ 指定 MLflow 追蹤伺服器的 URI。
+ **run_id** *(str)* ─ MLflow 實驗運行的唯一標識符。
+ **text** *(str)* ─ Chest CT 影像文字報告中的文本。

輸出結果：
```
1. A 2 cm mass in the right upper lobe ( RUL ) , highly suspicious for primary lung cancer .
```