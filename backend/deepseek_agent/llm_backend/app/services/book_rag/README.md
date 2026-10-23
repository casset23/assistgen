# Book RAG 分块参数说明

本目录下的 `chunking_config.json` 用于控制教材类 PDF 的分块行为。


```json
{
  "target_chunk_length": 1100,
  "max_chunk_length": 1500,
  "chunk_soft_limit": 1620,
  "chunk_overlap": 120,
  "min_chunk_length": 320
}
```

## 参数说明

### `target_chunk_length`
目标 chunk 长度。达到这个长度附近，并且文本边界合适时，会优先切分。

- 调大：chunk 数量变少，单块更长
- 调小：chunk 数量变多，单块更细
- 通用建议：`1000 ~ 1200`

### `max_chunk_length`
chunk 的硬上限。超过这个长度后，原则上不再继续追加内容。

- 调大：更容易保留大段上下文
- 调小：更容易切开长段
- 通用建议：`1400 ~ 1700`
- 不建议超过：`1800`

### `chunk_soft_limit`
chunk 的软上限。允许在“补半句话”时略微超过硬上限，但不会无限继续拼接。

- 通常设置为 `max_chunk_length + 80 ~ 180`
- 如果经常出现“差半句就断开”，可以略微调大
- 如果又开始过度合并，就调小

### `chunk_overlap`
相邻 chunk 的重叠长度，用于增强检索时的上下文连续性。

- 调大：前后文衔接更强，但重复内容更多
- 调小：重复更少，但可能漏掉相邻上下文
- 通用建议：`80 ~ 150`

### `min_chunk_length`
最小理想 chunk 长度。太短的尾块更容易被并回前一块。

- 调大：减少碎块
- 调小：保留更细粒度的小块
- 通用建议：`250 ~ 400`

## 调参指南

### 1. chunk 太碎、数量太多
表现：`chunk数` 太多，平均长度偏小。

建议：
- 增大 `target_chunk_length`
- 适当增大 `min_chunk_length`

示例：

```json
{
  "target_chunk_length": 1250,
  "max_chunk_length": 1500,
  "chunk_soft_limit": 1620,
  "chunk_overlap": 120,
  "min_chunk_length": 380
}
```

### 2. chunk 太大、检索粒度太粗
表现：chunk 数量太少，一块里混入太多主题。

建议：
- 减小 `target_chunk_length`
- 适当减小 `max_chunk_length`

示例：

```json
{
  "target_chunk_length": 950,
  "max_chunk_length": 1350,
  "chunk_soft_limit": 1470,
  "chunk_overlap": 120,
  "min_chunk_length": 280
}
```

### 3. 检索时上下文衔接不够
表现：答案在相邻段，但召回片段不完整。

建议：
- 增大 `chunk_overlap`

示例：

```json
{
  "target_chunk_length": 1100,
  "max_chunk_length": 1500,
  "chunk_soft_limit": 1620,
  "chunk_overlap": 150,
  "min_chunk_length": 320
}
```

### 4. 末尾经常出现很短的碎块
表现：最后多出一小块，内容很短，可读性差。

建议：
- 增大 `min_chunk_length`

### 5. 经常差半句就切开
表现：chunk 末尾看起来经常只差一点点上下文。

建议：
- 略微增大 `chunk_soft_limit`
- 不要一次改太多，建议每次加 `40 ~ 80`

## 推荐使用方式

### 通用教材默认值
适用于大多数教材，先跑再看结果：

```json
{
  "target_chunk_length": 1100,
  "max_chunk_length": 1500,
  "chunk_soft_limit": 1620,
  "chunk_overlap": 120,
  "min_chunk_length": 320
}
```

### 偏精细检索
适合定义较多、公式较多、希望切细一点的教材：

```json
{
  "target_chunk_length": 950,
  "max_chunk_length": 1300,
  "chunk_soft_limit": 1420,
  "chunk_overlap": 140,
  "min_chunk_length": 280
}
```

### 偏稳健粗粒度
适合连续叙述较多的教材：

```json
{
  "target_chunk_length": 1250,
  "max_chunk_length": 1650,
  "chunk_soft_limit": 1780,
  "chunk_overlap": 120,
  "min_chunk_length": 360
}
```

## 实际建议

换教材时，建议按下面顺序调参：

1. `target_chunk_length`
2. `chunk_overlap`
3. `min_chunk_length`
4. `max_chunk_length`
5. `chunk_soft_limit`

尽量一次只改 1~2 个参数，先跑一次，再继续微调，不要同时大改全部参数。

