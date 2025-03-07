### 1. Create env

```
conda create --name msde python=3.11 -y
conda activate msde
pip install -r requirements.txt
```

### 2. Inference

```
python inference.py \
  --input_filepath <path_to_input_txt>  \
  --output_filepath <path_to_output_txt>  \
  --prompt_filepath <path_to_prompt_file>
```

