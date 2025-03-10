# 中文多技能对话评估基准

中文多技能对话评估基准——Multi-Skill Dialogue Evaluation Benchmark (MSDE)，包含1781个对话和21218个话语，覆盖了闲聊、知识对话、画像聊天和对话推荐等对话类型，为多技能对话的评估奠定了基础。

本链接公开了MSDE基准的完整数据集。数据集按照对话任务类型进行组织，在data目录下为每种任务类型建立了独立的子目录。各子目录采用与论文中一致的任务命名规范进行标识。每个任务子目录包含两个主要部分：（1）原始对话数据文件（.txt格式）；（2）多个大规模预训练语言模型在该任务上的性能评估结果(.xlsx文件)，其中包括模型生成结果文件以及相应的人工评估结果。这种结构化的数据组织方式便于研究者进行系统的性能比较和分析。



#### 目录结构

```bash
MSDE/
├── scripts/                     
│   └── inference.py  
│   └── requirement.txt 
│   └── readme.md           
├── data/                     
│   ├── CCF LUGE_DuConv/
│   │   ├── CCF LUGE_DuConv.txt
│   │   ├── baichuan.xlsx
│   │   ├── chatglm.xlsx
│   │   ├── llama.xlsx
│   │   └── qianwen.xlsx
│   ├── CCF LUGE_LCCC/
│   │   ├── CCF LUGE_LCCC.txt
│   │   ├── baichuan.xlsx
│   │   ├── chatglm.xlsx
│   │   ├── llama.xlsx
│   │   └── qianwen.xlsx
│   ├── CCF LUGE_DuRecDial/
│   │   ├── CCF LUGE_DuRecDial.txt
│   │   ├── baichuan.xlsx
│   │   ├── chatglm.xlsx
│   │   ├── llama.xlsx
│   │   └── qianwen.xlsx
│   ├── Lic 2021_CPC/
│   │   ├── Lic 2021_CPC.txt
│   │   ├── baichuan.xlsx
│   │   ├── chatglm.xlsx
│   │   ├── llama.xlsx
│   │   └── qianwen.xlsx
│   ├── Lic 2021_DuRecDial/
│   │   ├── Lic 2021_DuRecDial.txt
│   │   ├── baichuan.xlsx
│   │   ├── chatglm.xlsx
│   │   ├── llama.xlsx
│   │   └── qianwen.xlsx
│   └── Lic 2021_CPC/
│       ├── Lic 2021_CPC.txt
│       ├── baichuan.xlsx
│       ├── chatglm.xlsx
│       ├── llama.xlsx
│       └── qianwen.xlsx
└── README.md

```





