# financial_readability

Package that implements the BOFIR score - a readability measure for business and financial texts. The measure ranks a paragraph based on 399 textual features into 5 readability categories ranging from 1 - "very hard to read" to 5 - "easy to read". The measure is also available on a three category scale.

## Setup
Install lastest version from GitHub:
```
git clone https://github.com/grooof/financial_readability.git
cd financial_readability
pip install .
```

## Usage

```
>>> import financial_readability as fr

>>> test_data = (
    "This could be some complicated business text that contains a lot of difficult language, and is therefore poorly specified in classic readability formulas."
)

readability = financial_readability(test_data)

```


## List of Functions