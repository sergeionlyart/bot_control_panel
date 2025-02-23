#!/bin/bash
# сделать скроипт исполняеым chmod +x export_to_md.sh
# запустить скрипт ./export_to_md.sh 
# Укажите корневую директорию вашего проекта
PROJECT_DIR="/Users/sergejavdejcik/Library/Mobile Documents/com~apple~CloudDocs/code_2/Umnico_Widget_Test/bot_control_panel"

# Название выходного .md-файла
OUTPUT_FILE="project_bot_control_panel.md"

############################################
# 1. Выводим структуру проекта (tree)
#    Исключаем venv, .venv, __pycache__, .git,
#    а также bin, lib, include, pyvenv.cfg
############################################
echo "# Project Structure" > "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
tree "$PROJECT_DIR" -I "venv|\.venv|__pycache__|\.git|bin|lib|include|pyvenv.cfg" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

############################################
# 2. Заголовок для исходного кода
############################################
echo "# Source Code" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

############################################
# 3. Находим все .py-файлы, исключая те же каталоги
############################################
find "$PROJECT_DIR" \
  -path "*/venv/*" -prune -o \
  -path "*/.venv/*" -prune -o \
  -path "*/__pycache__/*" -prune -o \
  -path "*/.git/*" -prune -o \
  -path "*/bin/*" -prune -o \
  -path "*/lib/*" -prune -o \
  -path "*/include/*" -prune -o \
  -name "pyvenv.cfg" -prune -o \
  -name "*.py" -print |
while read file; do
  echo "## $file" >> "$OUTPUT_FILE"
  echo '```python' >> "$OUTPUT_FILE"
  cat "$file" >> "$OUTPUT_FILE"
  echo '```' >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
done

echo "Done. Created $OUTPUT_FILE."