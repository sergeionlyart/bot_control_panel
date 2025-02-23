project/
  ├─ main.py
  ├─ tab1/
  │   ├─ __init__.py
  │   └─ tab1_code.py
  ├─ tab2/
  │   ├─ __init__.py
  │   └─ tab2_code.py
  └─ tab3/
      ├─ __init__.py
      └─ tab3_code.py



"""
    Вызывает get_top_3_documents, а затем преобразует результат в JSON
    нужного вам формата:
      {
        "documents": [
          {
            "id": "<doc_id>",
            "text": "<Исходный текст>", // вопрос пользователя -клиента 
            "response": "<system_response>"// эталонный ответ бота 
            embedding ---- вектора 
          },
          ...
        ]
      }
    """      