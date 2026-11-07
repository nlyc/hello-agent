# M8 检索与 RAG 评估报告

## 评估设置

- 语料：4 篇旅行知识，共 13 个 chunk
- 评估集：`data/eval/travel_qa.jsonl`，9 条单跳问题
- 检索：BM25 单路 / Hashing 稠密 + BM25 混合（RRF）/ 可选 FastEmbed + BM25
- 生成方式：`py -3.14 scripts/eval_rag.py --out docs/m8-memory-rag-eval.md`

跳过的后端：fastembed（未安装，跳过语义模式）

## 召回命中率

| 后端 | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| bm25 | 0.78 | 1.00 | 1.00 | 0.89 |
| hashing | 0.78 | 1.00 | 1.00 | 0.89 |

## 回答层指标（带 RAG vs 不带 RAG）

| 后端 | 事实命中率 | 引用覆盖率 | 拒答率 | 平均完整度(1–5) |
| --- | --- | --- | --- | --- |
| bm25 | 0.89 | 1.00 | 0.00 | 4.67 |
| hashing | 0.89 | 1.00 | 0.00 | 4.67 |
| no_rag | 0.00 | 0.00 | 0.00 | 1.00 |

## 逐题对照（后端：hashing）

| 问题 | 标准答案 | 命中排名 | 不带 RAG | 带 RAG（摘录） | 事实命中 |
| --- | --- | --- | --- | --- | --- |
| 故宫周几闭馆？ | 周一闭馆 | 1 | 推荐当地热门景点，建议出发前查询开放时间并关注天气。 | 故宫博物院位于北京中轴线，旺季门票六十元。开放时间为八点半到十七点，周一闭馆，法定节假日除外。故宫大部分展馆位于室内，雨天可以正常参观。 | 是 |
| 颐和园适合爬坡吗？ | 适合不爬坡的游客 | 1 | 推荐当地热门景点，建议出发前查询开放时间并关注天气。 | 颐和园以昆明湖和万寿山为主体，适合半日游。东宫门到长廊一线地势平缓，适合不爬坡的游客；登万寿山需要连续爬台阶，腿脚不便者建议绕行。 | 是 |
| 北京下雨天可以去哪些室内景点？ | 中国国家博物馆 | 2 | 推荐当地热门景点，建议出发前查询开放时间并关注天气。 | 北京雨天优先选择室内景点，例如故宫博物院、中国国家博物馆和首都博物馆。香山、八达岭长城等户外景点在山路湿滑时应避免前往。 | 是 |
| 香山红叶什么时候最好看？ | 十月中旬到十一月初 | 1 | 推荐当地热门景点，建议出发前查询开放时间并关注天气。 | 香山公园以秋季红叶闻名，最佳观赏期是十月中旬到十一月初。登顶需要连续爬坡约一小时，雨天山路湿滑，不建议前往。 | 是 |
| 外滩夜景几点最好？ | 晚上七点至九点 | 2 | 推荐当地热门景点，建议出发前查询开放时间并关注天气。 | 上海雨天推荐上海博物馆和上海科技馆等室内场馆。外滩夜景可以改到次日晴天再去看。 | 否 |
| 上海雨天适合去哪里？ | 上海博物馆 | 1 | 推荐当地热门景点，建议出发前查询开放时间并关注天气。 | 上海雨天推荐上海博物馆和上海科技馆等室内场馆。外滩夜景可以改到次日晴天再去看。 | 是 |
| 西湖适合骑行吗？ | 适合环湖骑行 | 1 | 推荐当地热门景点，建议出发前查询开放时间并关注天气。 | 西湖环湖道路约十五公里，适合环湖骑行，也可以乘游船。苏堤和白堤是经典步行路线，环湖一圈大约需要两到三小时。 | 是 |
| 成都大熊猫基地什么时候去最好？ | 早上开园时前往 | 1 | 推荐当地热门景点，建议出发前查询开放时间并关注天气。 | 成都大熊猫繁育研究基地位于北郊斧头山，建议早上开园时前往。上午是熊猫活动高峰，中午以后多数熊猫会休息。 | 是 |
| 都江堰雨天能去吗？ | 改期 | 1 | 推荐当地热门景点，建议出发前查询开放时间并关注天气。 | 成都雨天推荐成都博物馆和四川博物院等室内场馆。都江堰行程建议改期，山区道路在雨天较为湿滑。 | 是 |

## 原始结果

<details>
<summary>展开 JSON</summary>

```json
{
  "cases": [
    {
      "id": "qa_01",
      "question": "故宫周几闭馆？",
      "gold_chunk_ids": [
        "chunk_c492627a87bd4d73"
      ],
      "gold_answer": "周一闭馆",
      "city": "北京",
      "tags": [
        "single_hop"
      ]
    },
    {
      "id": "qa_02",
      "question": "颐和园适合爬坡吗？",
      "gold_chunk_ids": [
        "chunk_057f942b7ff24a2d"
      ],
      "gold_answer": "适合不爬坡的游客",
      "city": "北京",
      "tags": [
        "single_hop"
      ]
    },
    {
      "id": "qa_03",
      "question": "北京下雨天可以去哪些室内景点？",
      "gold_chunk_ids": [
        "chunk_3c0ee3d8b6ca197e"
      ],
      "gold_answer": "中国国家博物馆",
      "city": "北京",
      "tags": [
        "single_hop"
      ]
    },
    {
      "id": "qa_04",
      "question": "香山红叶什么时候最好看？",
      "gold_chunk_ids": [
        "chunk_a7dc84813cba66a7"
      ],
      "gold_answer": "十月中旬到十一月初",
      "city": "北京",
      "tags": [
        "single_hop"
      ]
    },
    {
      "id": "qa_05",
      "question": "外滩夜景几点最好？",
      "gold_chunk_ids": [
        "chunk_da6a5fcd8f18407b"
      ],
      "gold_answer": "晚上七点至九点",
      "city": "上海",
      "tags": [
        "single_hop"
      ]
    },
    {
      "id": "qa_06",
      "question": "上海雨天适合去哪里？",
      "gold_chunk_ids": [
        "chunk_3429f21bf4a8211f"
      ],
      "gold_answer": "上海博物馆",
      "city": "上海",
      "tags": [
        "single_hop"
      ]
    },
    {
      "id": "qa_07",
      "question": "西湖适合骑行吗？",
      "gold_chunk_ids": [
        "chunk_e09f1372fdd42243"
      ],
      "gold_answer": "适合环湖骑行",
      "city": "杭州",
      "tags": [
        "single_hop"
      ]
    },
    {
      "id": "qa_08",
      "question": "成都大熊猫基地什么时候去最好？",
      "gold_chunk_ids": [
        "chunk_7ebe009c2bd12879"
      ],
      "gold_answer": "早上开园时前往",
      "city": "成都",
      "tags": [
        "single_hop"
      ]
    },
    {
      "id": "qa_09",
      "question": "都江堰雨天能去吗？",
      "gold_chunk_ids": [
        "chunk_7165131a7378cac8"
      ],
      "gold_answer": "改期",
      "city": "成都",
      "tags": [
        "single_hop"
      ]
    }
  ],
  "results": [
    {
      "backend": "bm25",
      "embedding_fingerprint": "bm25",
      "recall": {
        "recall@1": 0.7778,
        "recall@3": 1.0,
        "recall@5": 1.0
      },
      "mrr": 0.8889,
      "fact_rate": 0.8889,
      "citation_rate": 1.0,
      "refusal_rate": 0.0,
      "completeness": 4.67,
      "details": [
        {
          "id": "qa_01",
          "question": "故宫周几闭馆？",
          "gold_answer": "周一闭馆",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_c492627a87bd4d73",
            "chunk_3c0ee3d8b6ca197e"
          ],
          "answer": "故宫博物院位于北京中轴线，旺季门票六十元。开放时间为八点半到十七点，周一闭馆，法定节假日除外。故宫大部分展馆位于室内，雨天可以正常参观。\n依据：北京旅行知识 北京旅行知识 / 故宫",
          "used_citation": "北京旅行知识 北京旅行知识 / 故宫",
          "citations": [
            "北京旅行知识 北京旅行知识 / 故宫",
            "北京旅行知识 北京旅行知识 / 北京雨天备选"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_02",
          "question": "颐和园适合爬坡吗？",
          "gold_answer": "适合不爬坡的游客",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_057f942b7ff24a2d",
            "chunk_a7dc84813cba66a7",
            "chunk_8654b50b119e88ca",
            "chunk_da6a5fcd8f18407b",
            "chunk_e09f1372fdd42243"
          ],
          "answer": "颐和园以昆明湖和万寿山为主体，适合半日游。东宫门到长廊一线地势平缓，适合不爬坡的游客；登万寿山需要连续爬台阶，腿脚不便者建议绕行。\n依据：北京旅行知识 北京旅行知识 / 颐和园",
          "used_citation": "北京旅行知识 北京旅行知识 / 颐和园",
          "citations": [
            "北京旅行知识 北京旅行知识 / 颐和园",
            "北京旅行知识 北京旅行知识 / 香山",
            "成都旅行知识 成都旅行知识 / 都江堰",
            "上海旅行知识 上海旅行知识 / 外滩",
            "杭州旅行知识 杭州旅行知识 / 西湖"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_03",
          "question": "北京下雨天可以去哪些室内景点？",
          "gold_answer": "中国国家博物馆",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_3c0ee3d8b6ca197e",
            "chunk_c492627a87bd4d73",
            "chunk_029273fe93dc8869",
            "chunk_3429f21bf4a8211f",
            "chunk_7165131a7378cac8"
          ],
          "answer": "北京雨天优先选择室内景点，例如故宫博物院、中国国家博物馆和首都博物馆。香山、八达岭长城等户外景点在山路湿滑时应避免前往。\n依据：北京旅行知识 北京旅行知识 / 北京雨天备选",
          "used_citation": "北京旅行知识 北京旅行知识 / 北京雨天备选",
          "citations": [
            "北京旅行知识 北京旅行知识 / 北京雨天备选",
            "北京旅行知识 北京旅行知识 / 故宫",
            "杭州旅行知识 杭州旅行知识 / 杭州雨天备选",
            "上海旅行知识 上海旅行知识 / 上海雨天备选",
            "成都旅行知识 成都旅行知识 / 成都雨天备选"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_04",
          "question": "香山红叶什么时候最好看？",
          "gold_answer": "十月中旬到十一月初",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_a7dc84813cba66a7",
            "chunk_3c0ee3d8b6ca197e"
          ],
          "answer": "香山公园以秋季红叶闻名，最佳观赏期是十月中旬到十一月初。登顶需要连续爬坡约一小时，雨天山路湿滑，不建议前往。\n依据：北京旅行知识 北京旅行知识 / 香山",
          "used_citation": "北京旅行知识 北京旅行知识 / 香山",
          "citations": [
            "北京旅行知识 北京旅行知识 / 香山",
            "北京旅行知识 北京旅行知识 / 北京雨天备选"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_05",
          "question": "外滩夜景几点最好？",
          "gold_answer": "晚上七点至九点",
          "rank": 2,
          "top_chunk_ids": [
            "chunk_3429f21bf4a8211f",
            "chunk_da6a5fcd8f18407b"
          ],
          "answer": "上海雨天推荐上海博物馆和上海科技馆等室内场馆。外滩夜景可以改到次日晴天再去看。\n依据：上海旅行知识 上海旅行知识 / 上海雨天备选",
          "used_citation": "上海旅行知识 上海旅行知识 / 上海雨天备选",
          "citations": [
            "上海旅行知识 上海旅行知识 / 上海雨天备选",
            "上海旅行知识 上海旅行知识 / 外滩"
          ],
          "fact_hit": false,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_06",
          "question": "上海雨天适合去哪里？",
          "gold_answer": "上海博物馆",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_3429f21bf4a8211f",
            "chunk_da6a5fcd8f18407b",
            "chunk_057f942b7ff24a2d",
            "chunk_1774e4fc288f980a",
            "chunk_8654b50b119e88ca"
          ],
          "answer": "上海雨天推荐上海博物馆和上海科技馆等室内场馆。外滩夜景可以改到次日晴天再去看。\n依据：上海旅行知识 上海旅行知识 / 上海雨天备选",
          "used_citation": "上海旅行知识 上海旅行知识 / 上海雨天备选",
          "citations": [
            "上海旅行知识 上海旅行知识 / 上海雨天备选",
            "上海旅行知识 上海旅行知识 / 外滩",
            "北京旅行知识 北京旅行知识 / 颐和园",
            "上海旅行知识 上海旅行知识 / 豫园",
            "成都旅行知识 成都旅行知识 / 都江堰"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_07",
          "question": "西湖适合骑行吗？",
          "gold_answer": "适合环湖骑行",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_e09f1372fdd42243",
            "chunk_0ba14b9a3dc89c5e",
            "chunk_029273fe93dc8869",
            "chunk_057f942b7ff24a2d",
            "chunk_8654b50b119e88ca"
          ],
          "answer": "西湖环湖道路约十五公里，适合环湖骑行，也可以乘游船。苏堤和白堤是经典步行路线，环湖一圈大约需要两到三小时。\n依据：杭州旅行知识 杭州旅行知识 / 西湖",
          "used_citation": "杭州旅行知识 杭州旅行知识 / 西湖",
          "citations": [
            "杭州旅行知识 杭州旅行知识 / 西湖",
            "杭州旅行知识 杭州旅行知识 / 灵隐寺",
            "杭州旅行知识 杭州旅行知识 / 杭州雨天备选",
            "北京旅行知识 北京旅行知识 / 颐和园",
            "成都旅行知识 成都旅行知识 / 都江堰"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_08",
          "question": "成都大熊猫基地什么时候去最好？",
          "gold_answer": "早上开园时前往",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_7ebe009c2bd12879",
            "chunk_7165131a7378cac8",
            "chunk_8654b50b119e88ca"
          ],
          "answer": "成都大熊猫繁育研究基地位于北郊斧头山，建议早上开园时前往。上午是熊猫活动高峰，中午以后多数熊猫会休息。\n依据：成都旅行知识 成都旅行知识 / 大熊猫基地",
          "used_citation": "成都旅行知识 成都旅行知识 / 大熊猫基地",
          "citations": [
            "成都旅行知识 成都旅行知识 / 大熊猫基地",
            "成都旅行知识 成都旅行知识 / 成都雨天备选",
            "成都旅行知识 成都旅行知识 / 都江堰"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_09",
          "question": "都江堰雨天能去吗？",
          "gold_answer": "改期",
          "rank": 2,
          "top_chunk_ids": [
            "chunk_8654b50b119e88ca",
            "chunk_7165131a7378cac8",
            "chunk_3429f21bf4a8211f",
            "chunk_029273fe93dc8869",
            "chunk_3c0ee3d8b6ca197e"
          ],
          "answer": "成都雨天推荐成都博物馆和四川博物院等室内场馆。都江堰行程建议改期，山区道路在雨天较为湿滑。\n依据：成都旅行知识 成都旅行知识 / 成都雨天备选",
          "used_citation": "成都旅行知识 成都旅行知识 / 成都雨天备选",
          "citations": [
            "成都旅行知识 成都旅行知识 / 都江堰",
            "成都旅行知识 成都旅行知识 / 成都雨天备选",
            "上海旅行知识 上海旅行知识 / 上海雨天备选",
            "杭州旅行知识 杭州旅行知识 / 杭州雨天备选",
            "北京旅行知识 北京旅行知识 / 北京雨天备选"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        }
      ]
    },
    {
      "backend": "hashing",
      "embedding_fingerprint": "hashing:char-ngram:512",
      "recall": {
        "recall@1": 0.7778,
        "recall@3": 1.0,
        "recall@5": 1.0
      },
      "mrr": 0.8889,
      "fact_rate": 0.8889,
      "citation_rate": 1.0,
      "refusal_rate": 0.0,
      "completeness": 4.67,
      "details": [
        {
          "id": "qa_01",
          "question": "故宫周几闭馆？",
          "gold_answer": "周一闭馆",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_c492627a87bd4d73",
            "chunk_3c0ee3d8b6ca197e"
          ],
          "answer": "故宫博物院位于北京中轴线，旺季门票六十元。开放时间为八点半到十七点，周一闭馆，法定节假日除外。故宫大部分展馆位于室内，雨天可以正常参观。\n依据：北京旅行知识 北京旅行知识 / 故宫",
          "used_citation": "北京旅行知识 北京旅行知识 / 故宫",
          "citations": [
            "北京旅行知识 北京旅行知识 / 故宫",
            "北京旅行知识 北京旅行知识 / 北京雨天备选"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_02",
          "question": "颐和园适合爬坡吗？",
          "gold_answer": "适合不爬坡的游客",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_057f942b7ff24a2d",
            "chunk_8654b50b119e88ca",
            "chunk_a7dc84813cba66a7",
            "chunk_da6a5fcd8f18407b",
            "chunk_e09f1372fdd42243"
          ],
          "answer": "颐和园以昆明湖和万寿山为主体，适合半日游。东宫门到长廊一线地势平缓，适合不爬坡的游客；登万寿山需要连续爬台阶，腿脚不便者建议绕行。\n依据：北京旅行知识 北京旅行知识 / 颐和园",
          "used_citation": "北京旅行知识 北京旅行知识 / 颐和园",
          "citations": [
            "北京旅行知识 北京旅行知识 / 颐和园",
            "成都旅行知识 成都旅行知识 / 都江堰",
            "北京旅行知识 北京旅行知识 / 香山",
            "上海旅行知识 上海旅行知识 / 外滩",
            "杭州旅行知识 杭州旅行知识 / 西湖"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_03",
          "question": "北京下雨天可以去哪些室内景点？",
          "gold_answer": "中国国家博物馆",
          "rank": 2,
          "top_chunk_ids": [
            "chunk_029273fe93dc8869",
            "chunk_3c0ee3d8b6ca197e",
            "chunk_c492627a87bd4d73",
            "chunk_3429f21bf4a8211f",
            "chunk_7165131a7378cac8"
          ],
          "answer": "北京雨天优先选择室内景点，例如故宫博物院、中国国家博物馆和首都博物馆。香山、八达岭长城等户外景点在山路湿滑时应避免前往。\n依据：北京旅行知识 北京旅行知识 / 北京雨天备选",
          "used_citation": "北京旅行知识 北京旅行知识 / 北京雨天备选",
          "citations": [
            "杭州旅行知识 杭州旅行知识 / 杭州雨天备选",
            "北京旅行知识 北京旅行知识 / 北京雨天备选",
            "北京旅行知识 北京旅行知识 / 故宫",
            "上海旅行知识 上海旅行知识 / 上海雨天备选",
            "成都旅行知识 成都旅行知识 / 成都雨天备选"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_04",
          "question": "香山红叶什么时候最好看？",
          "gold_answer": "十月中旬到十一月初",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_a7dc84813cba66a7",
            "chunk_057f942b7ff24a2d",
            "chunk_3c0ee3d8b6ca197e",
            "chunk_7ebe009c2bd12879",
            "chunk_c492627a87bd4d73"
          ],
          "answer": "香山公园以秋季红叶闻名，最佳观赏期是十月中旬到十一月初。登顶需要连续爬坡约一小时，雨天山路湿滑，不建议前往。\n依据：北京旅行知识 北京旅行知识 / 香山",
          "used_citation": "北京旅行知识 北京旅行知识 / 香山",
          "citations": [
            "北京旅行知识 北京旅行知识 / 香山",
            "北京旅行知识 北京旅行知识 / 颐和园",
            "北京旅行知识 北京旅行知识 / 北京雨天备选",
            "成都旅行知识 成都旅行知识 / 大熊猫基地",
            "北京旅行知识 北京旅行知识 / 故宫"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_05",
          "question": "外滩夜景几点最好？",
          "gold_answer": "晚上七点至九点",
          "rank": 2,
          "top_chunk_ids": [
            "chunk_3429f21bf4a8211f",
            "chunk_da6a5fcd8f18407b"
          ],
          "answer": "上海雨天推荐上海博物馆和上海科技馆等室内场馆。外滩夜景可以改到次日晴天再去看。\n依据：上海旅行知识 上海旅行知识 / 上海雨天备选",
          "used_citation": "上海旅行知识 上海旅行知识 / 上海雨天备选",
          "citations": [
            "上海旅行知识 上海旅行知识 / 上海雨天备选",
            "上海旅行知识 上海旅行知识 / 外滩"
          ],
          "fact_hit": false,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_06",
          "question": "上海雨天适合去哪里？",
          "gold_answer": "上海博物馆",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_3429f21bf4a8211f",
            "chunk_7165131a7378cac8",
            "chunk_057f942b7ff24a2d",
            "chunk_029273fe93dc8869",
            "chunk_e09f1372fdd42243"
          ],
          "answer": "上海雨天推荐上海博物馆和上海科技馆等室内场馆。外滩夜景可以改到次日晴天再去看。\n依据：上海旅行知识 上海旅行知识 / 上海雨天备选",
          "used_citation": "上海旅行知识 上海旅行知识 / 上海雨天备选",
          "citations": [
            "上海旅行知识 上海旅行知识 / 上海雨天备选",
            "成都旅行知识 成都旅行知识 / 成都雨天备选",
            "北京旅行知识 北京旅行知识 / 颐和园",
            "杭州旅行知识 杭州旅行知识 / 杭州雨天备选",
            "杭州旅行知识 杭州旅行知识 / 西湖"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_07",
          "question": "西湖适合骑行吗？",
          "gold_answer": "适合环湖骑行",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_e09f1372fdd42243",
            "chunk_8654b50b119e88ca",
            "chunk_057f942b7ff24a2d",
            "chunk_029273fe93dc8869",
            "chunk_da6a5fcd8f18407b"
          ],
          "answer": "西湖环湖道路约十五公里，适合环湖骑行，也可以乘游船。苏堤和白堤是经典步行路线，环湖一圈大约需要两到三小时。\n依据：杭州旅行知识 杭州旅行知识 / 西湖",
          "used_citation": "杭州旅行知识 杭州旅行知识 / 西湖",
          "citations": [
            "杭州旅行知识 杭州旅行知识 / 西湖",
            "成都旅行知识 成都旅行知识 / 都江堰",
            "北京旅行知识 北京旅行知识 / 颐和园",
            "杭州旅行知识 杭州旅行知识 / 杭州雨天备选",
            "上海旅行知识 上海旅行知识 / 外滩"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_08",
          "question": "成都大熊猫基地什么时候去最好？",
          "gold_answer": "早上开园时前往",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_7ebe009c2bd12879",
            "chunk_7165131a7378cac8",
            "chunk_0ba14b9a3dc89c5e",
            "chunk_8654b50b119e88ca"
          ],
          "answer": "成都大熊猫繁育研究基地位于北郊斧头山，建议早上开园时前往。上午是熊猫活动高峰，中午以后多数熊猫会休息。\n依据：成都旅行知识 成都旅行知识 / 大熊猫基地",
          "used_citation": "成都旅行知识 成都旅行知识 / 大熊猫基地",
          "citations": [
            "成都旅行知识 成都旅行知识 / 大熊猫基地",
            "成都旅行知识 成都旅行知识 / 成都雨天备选",
            "杭州旅行知识 杭州旅行知识 / 灵隐寺",
            "成都旅行知识 成都旅行知识 / 都江堰"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        },
        {
          "id": "qa_09",
          "question": "都江堰雨天能去吗？",
          "gold_answer": "改期",
          "rank": 1,
          "top_chunk_ids": [
            "chunk_7165131a7378cac8",
            "chunk_8654b50b119e88ca",
            "chunk_3c0ee3d8b6ca197e",
            "chunk_029273fe93dc8869",
            "chunk_3429f21bf4a8211f"
          ],
          "answer": "成都雨天推荐成都博物馆和四川博物院等室内场馆。都江堰行程建议改期，山区道路在雨天较为湿滑。\n依据：成都旅行知识 成都旅行知识 / 成都雨天备选",
          "used_citation": "成都旅行知识 成都旅行知识 / 成都雨天备选",
          "citations": [
            "成都旅行知识 成都旅行知识 / 成都雨天备选",
            "成都旅行知识 成都旅行知识 / 都江堰",
            "北京旅行知识 北京旅行知识 / 北京雨天备选",
            "杭州旅行知识 杭州旅行知识 / 杭州雨天备选",
            "上海旅行知识 上海旅行知识 / 上海雨天备选"
          ],
          "fact_hit": true,
          "citation_hit": true,
          "no_answer": false
        }
      ]
    }
  ],
  "no_rag": {
    "backend": "no_rag",
    "embedding_fingerprint": "-",
    "recall": {
      "recall@1": 0.0,
      "recall@3": 0.0,
      "recall@5": 0.0
    },
    "mrr": 0.0,
    "fact_rate": 0.0,
    "citation_rate": 0.0,
    "refusal_rate": 0.0,
    "completeness": 1.0,
    "details": [
      {
        "id": "qa_01",
        "question": "故宫周几闭馆？",
        "gold_answer": "周一闭馆",
        "rank": null,
        "answer": "推荐当地热门景点，建议出发前查询开放时间并关注天气。",
        "citations": [],
        "fact_hit": false,
        "citation_hit": false,
        "no_answer": false
      },
      {
        "id": "qa_02",
        "question": "颐和园适合爬坡吗？",
        "gold_answer": "适合不爬坡的游客",
        "rank": null,
        "answer": "推荐当地热门景点，建议出发前查询开放时间并关注天气。",
        "citations": [],
        "fact_hit": false,
        "citation_hit": false,
        "no_answer": false
      },
      {
        "id": "qa_03",
        "question": "北京下雨天可以去哪些室内景点？",
        "gold_answer": "中国国家博物馆",
        "rank": null,
        "answer": "推荐当地热门景点，建议出发前查询开放时间并关注天气。",
        "citations": [],
        "fact_hit": false,
        "citation_hit": false,
        "no_answer": false
      },
      {
        "id": "qa_04",
        "question": "香山红叶什么时候最好看？",
        "gold_answer": "十月中旬到十一月初",
        "rank": null,
        "answer": "推荐当地热门景点，建议出发前查询开放时间并关注天气。",
        "citations": [],
        "fact_hit": false,
        "citation_hit": false,
        "no_answer": false
      },
      {
        "id": "qa_05",
        "question": "外滩夜景几点最好？",
        "gold_answer": "晚上七点至九点",
        "rank": null,
        "answer": "推荐当地热门景点，建议出发前查询开放时间并关注天气。",
        "citations": [],
        "fact_hit": false,
        "citation_hit": false,
        "no_answer": false
      },
      {
        "id": "qa_06",
        "question": "上海雨天适合去哪里？",
        "gold_answer": "上海博物馆",
        "rank": null,
        "answer": "推荐当地热门景点，建议出发前查询开放时间并关注天气。",
        "citations": [],
        "fact_hit": false,
        "citation_hit": false,
        "no_answer": false
      },
      {
        "id": "qa_07",
        "question": "西湖适合骑行吗？",
        "gold_answer": "适合环湖骑行",
        "rank": null,
        "answer": "推荐当地热门景点，建议出发前查询开放时间并关注天气。",
        "citations": [],
        "fact_hit": false,
        "citation_hit": false,
        "no_answer": false
      },
      {
        "id": "qa_08",
        "question": "成都大熊猫基地什么时候去最好？",
        "gold_answer": "早上开园时前往",
        "rank": null,
        "answer": "推荐当地热门景点，建议出发前查询开放时间并关注天气。",
        "citations": [],
        "fact_hit": false,
        "citation_hit": false,
        "no_answer": false
      },
      {
        "id": "qa_09",
        "question": "都江堰雨天能去吗？",
        "gold_answer": "改期",
        "rank": null,
        "answer": "推荐当地热门景点，建议出发前查询开放时间并关注天气。",
        "citations": [],
        "fact_hit": false,
        "citation_hit": false,
        "no_answer": false
      }
    ]
  },
  "skipped": [
    "fastembed（未安装，跳过语义模式）"
  ]
}
```

</details>
