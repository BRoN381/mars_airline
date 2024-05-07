﻿# mars_airline
 
## setup
create conda env
```
cd path/to/mars_airline
pip install -r requirements.txt
```

## run main.py
```
cd path/to/mars_airline
uvicorn main:app --reload
```

## api
+ get_response
  requires JSON format with a 'question' item
