

> Written by Russell Gu

[TOC]

# Intro to Python

Python is an interpretive language.

> Getting your current working directory
> ```python
> import os
> os.getcwd()
> ``` 

## Python Basics

Python is **case-sensitive**.

### Functions

A very simplistic function:

```python
def test(A, B):
	C = A + B
	return C
```

Function can also return multiple values.

```python
def test(A, B):
	C = A + B
	D = A * B
	return C, D

# To assign the function's outcome

C, D = test(A, B)
```

Assignment in this case is also simple.

[List Operations](https://www.decalage.info/en/python/print_list)

### Multi-line Strings

```python
top_delinq = sqlContext.sql('''
select * from test_table t1
where t1.trans_dt = ''' 
+ str(trans_dt.collect()[0][0]) + 
''' and t1.states IN ('NY', 'NJ', 'MA') and t1.trans_amt >= 200000'''
)
```

### eval()

> The eval function lets a python program run python code within itself.

```python
>>> x = 1
>>> eval('x + 1')
2
>>> eval('x')
1
```
[Eval()](https://stackoverflow.com/questions/9383740/what-does-pythons-eval-do)

# Python Object Types

## List

List in Python is **mutable** as oppose to numpy arrays. This might cause unintended behavior or errors.

Print items:
```python
>>> mylist = ['spam', 'ham', 'eggs']
>>> print ', '.join(mylist)
spam, ham, eggs
```

Similarly, we can use:

```python
print '\n'.join(mylist)
```

Numeric List:
```python
[In]
list_of_ints = [80, 443, 8080, 8081]
print str(list_of_ints).strip('[]')

[Out]
80, 443, 8080, 8081
```

### Modifying a list

Because list is mutable, the operation will remove the item from the list. All copies of the list will also be modified in the exact same way.

```python
a.pop()		# removing the last item from the list
del a[-1]	# removing the last item from the list
```

### Slicing a list

```python
a[start:end] # items start through end-1
a[start:]    # items start through the rest of the array
a[:end]      # items from the beginning through end-1
a[:]         # a copy of the whole array
```

https://stackoverflow.com/questions/509211/understanding-pythons-slice-notation


## Dictionary

### Defining dictionary

```python
>>> dict_test = {'a':2, 'b':3}
# adding items to dictionary
>>> dict_test['c'] = 4
>>> dict_test
{'a':2, 'b':3, 'c':4}
```

#### Apply functions to list

##### Using Map()
```python
# Create a  lambda function
capitalizer = lambda x: x.upper()
# map function to the list and return a list 
regimentNamesCapitalized_m = list(map(capitalizer, regimentNames));
```
##### Using list comprehension
```python
regimentNamesCapitalized_l = [x.upper() for x in regimentNames];
```



## Data Type Manipulation

### Category variable

#### Converting column to categorical variable using `astype`

> This method sometimes does not work.

Setting a variable to be categorical. This will affect the ordering of the values when the data is presented in the pivot table.

```python
df['gender'] = df['gender'].astype('category', categories=['Male', 'Female'])
```

#### Converting column to categorical variable using `pd.Categorical`

```python
titanic['class'] = pd.Categorical(titanic['class'], ordered=True, categories=["First Class", "Second Class", "Nornaml Class"])
```

# NumPy

## Array

### Array Operation

#### `append()`
```python
>>> a = [1, 2]
>>> b = [3, 4]
>>> np.append(a, b)
array([1,2,3,4])
```

# Pandas

## Pandas Data Structure

There are two basic types of Pandas data structure: DataFrame and DataSeries

### Dataframe

> *Tips*
> 
> `.shape()` gives the dimension of the dataframe.
> `.columns` gives the column names

#### Slicing/select data

By column names
```python
df.loc[:, ['A', 'B']]
```

#### Subset

Subsetting data and conditional assignment can be safely done by using `.loc()`. Say that we want to create a variable `score` and assigned everyone in the first class a score of 5.

```python
titanic.loc[titanic['class'] == 'First Class', "score"] = 5
```

### Series

#### `.isin()`

`.isin()` will return Boolean on a Pandas series when giving an array.

```python

```

## Pandas File I/O

### Importing Data

#### Importing CSV file

`dtype`:  force datatype when importing
`usecols`: specify columns to import

```python
df = pd.read_csv('titanic.csv', 
			# forcing data to be str
			dtype={'passenger_id':str}, 
			# only import these columns
			usecols = ['passenger_id', 'nationality', 'price']) 
```


#### Importing CSV file as dictionary:

```python
import csv
import json

csvfile = open('file.csv', 'r')

fieldnames = ("FirstName","LastName","IDNumber","Message")
reader = csv.DictReader(csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')
```

#### Importing Fixed Width file

Importing with fixed width files. [Documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_fwf.html)

```python
df = pd.read_fwf('data/public_database/FedACHdir.txt', 
                                     widths = [9, 1, 9, 1, 6, 9, 36, 36, 20, 2, 5, 4, 3, 3, 4, 1, 1, 5], 
                                     header = None, 
                                     names = ['routing_number', 'office_code', 'servicing_frb_number', 'record_type_code', 'change_date', 'new_routing_number', 'customer_name', 'address', 'city', 'state_code', 'zipcode', 'zipcode_extension', 'telephone_area_code', 'telephone_prefix_number', 'telephone_suffix_number', 'institution_status_code', 'data_view_code', 'filler'],
                                    converters = {'routing_number': str})
```
`width` specifies the width of each columns.
`header` default value is 0. If header does not exist, it needs to be set to **None**.
`names` passes a array of names as column headers.
`converters` acts as `dtype` and preserve the data type of a field.

#### Importing csv as JSON

https://overlaid.net/2016/02/04/convert-a-csv-to-a-dictionary-in-python/

### Exporting

#### to CSV

Dataframe objects have `.to_csv` method ([Documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html)). By default, the method will preserve index (row names) when exporting. Setting `index = False` will take care of this problem.

```python
df.to_csv('file.csv', index = False)
```

#### to JSON

Write to JSON

```python
import json

with open('data.txt', 'w's) as outfile:  
    json.dump(data, outfile)
```

Read from JSON

```python

```

[Reading and Writing JSON to a File in Python](http://stackabuse.com/reading-and-writing-json-to-a-file-in-python/)

Write csv file from JSON
```python
dict_list = [{'name':'a', 'gender':'M'}, {'name':'b', 'gender':'F'}]

f = open('basic_info.csv','wb')
w = csv.DictWriter(f,dict_list[0].keys())	#dict_list is a list of dictionaries
w.writeheader()
w.writerows(dict_list)
f.close()
```

https://stackoverflow.com/questions/10373247/how-do-i-write-a-python-dictionary-to-a-csv-file

## Basic Pandas Methods & Functions

### Value Assignment

Assign the same value to the entire column.

```python
df['A'] = 1
```

#### Conditional value assignment

This is the equivalent of `ifelse` in R. It is also row rise operation.

```python
import numpy as np
q2['match_type'] = np.where(q2['DDA'].isnull(), 'No DDA', np.where(q2['DDA'].notnull() & q2['customer_name'].isnull(), 'No match', 'Match'))
```

### Column names

Below code will change the column name.

```python
df.rename(columns={"old_colname": "new_colname"})
```

You can also define the column names altogether.

```python
df.columns = ['a', 'b']
```

### Unique

```python
import pandas as pd
df['columnname'].unique
```

### Sort

```python
import pandas as pd
DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
```

### Join

```python
pd.merge(q2, frb_routing_number, left_on = 'DDA', right_on = 'routing_number', how = 'left')
```

### Append

`append` is a dataframe object method. It allows another object to be appended to the desired object.

```python
p = p1.append(p2).append(p3)
```

Different from `bind_rows` in R, the column names must be the same, or append will force new column to be created.

### Rename

```python
df.rename(columns={"new_routing_number": "routing_number"})
```

### Duplicate

#### Show Duplicate
```python
df.duplicated()
```

#### Drop Duplicate

```python
df.drop_duplicates()
```

#### Long to wide table 

```python
df.pivot(index='patient', columns='obs', values='score')
```

https://chrisalbon.com/python/pandas_long_to_wide.html

## String

Naive | Pandas
-------- | ---
`.lower()` | `.str.lower()`

`()` is needed. See debug log.


## Summarizing

Below query will allow you to calculate the percentage of the total of the values in the column.

```python
titanic['fare_pct'] = df.fare / df.fare.sum()
```

## Group

Group by in Pandas allows aggregation at the group level
```python
titanic.groupby('gender').agg({'fare':'sum'})
```

You can also group by at multiple levels by passing a list to group
```python
titanic.groupby(['gender', 'nationality', 'class']]).agg({'fare':'sum'})
```

### Slicing a group

After aggregating at multi-group level, you can slice the group while keeping the group structure.
```python
df_group = titanic.groupby(['gender', 'nationality', 'class']]).agg({'fare':'sum'})
df_group.loc[((slice(None), slice(None), slice('First Class')),:]
```

### Reset Index

`.reset_index()` will break the multi-index groups of data frame. You essentially have a long table with each multi-index group as a record.
```python
df_group.reset_index()
```

### Tabulate Data

#### Count by group

```python
port_2016q1.groupby('wo_ind_latest')[['contract_id']].count()
```

## Query

This is a feature similar to `select` in `dplyr` or `where` clause in SQL.

```python
fills.query('Symbol=="BUD US"')
```


### Local Variables

When you refer to a local variable in query, you have explicitly refer it as an variable. Query will be searching for the variable but will not be able for find it unless you use the `@` operator.

```python
my_symbol = 'BUD US'
fills.query('Symbol==my_symbol') # Error: my_symbol does not exist
fills.query('Symbol==@my_symbol') # works
```

[Explicitly refer to local variables](https://stackoverflow.com/questions/23974664/unable-to-query-a-local-variable-in-pandas-0-14-0)


## Pivot Table

[Documentation here.](https://pandas.pydata.org/pandas-docs/stable/reshaping.html#reshaping-and-pivot-tables)

```python
titanic.pivot_table(index='sex', columns='class',
                    aggfunc={'survived':sum, 'fare':'mean'})
```

When using pivot table, be careful that if the index has missing value, an `NaN` category will not be created automatically. Instead, you have to use a [work around](https://stackoverflow.com/questions/16860172/python-pandas-pivot-table-silently-drops-indices-with-nans).

> This part is not yet finished

```python
df['b'] = df['b'].fillna('error')
```

Sometimes you want pivot table to be arrange in the exact way you intended. For that reason, you might want to set categorical variables and arrange then in the desired order.

```python
q["loan_status"] = q["loan_status"].astype("category")
q["loan_status"].cat.set_categories(['CUR', '30A', '30D', '60D', 'NAC', 'WOF', 'CLS'],inplace=True)
```

#### Margins

In order for `margins = True` to work. It is safer to include **only the columns required** for the pivot table.

```python
(
	df.loc[:, ['nationality', 'class', 'price']] 
	# include only the required columns
	.pivot_table(index = 'nationality', columns = 'class', aggfunc = {'price':'sum'}, fill_value = 0, margins=True)
	.style
	.format({'tot_ar_bal': '${:,.2f}'})
)
```

### Pivot Table functions

Count: `count`
Sum: `sum`
Equivalent of `n_distinct` : `lambda x: len(x.unique())` OR `lambda x: x.nunique()`

Usage:
```python
q3.pivot_table(index = 'match_type', aggfunc = {'contract_nbr':lambda x: len(x.unique())})
```

### Style Pivot Table

`.style()` can be applied for dataframe. Further, `.format` can format any column with a given format.

```python
(
	df
	.pivot_table(index = ['event_name', 'action_type'], aggfunc = {'id':'count', 'end_bal':'sum'}).style
	.format({'end_bal_am': '${:,.2f}'})
)
```

Example formats:
`${:,.2f}`: dollar value with 2 decimal points
`{:.2f}%`: percentage with 2 decimal points

You can also format the entire table altogether.

```python
pivot.style.format('${:,.0f}')
```

[Styling Pivot Table](https://pandas.pydata.org/pandas-docs/stable/style.html)

## Piping

Pipe allows chaining several method consecutively, to perform a series of operations, each based on the result of the last operation. Pandas authors [prefer this method over subclass](https://pandas.pydata.org/pandas-docs/stable/internals.html#subclassing-pandas-data-structures) because of its simplicity.

Piping allows the use of functions in a way similar to class method. For example, to apply h, g, f in this order on `df` can be done using pipe in the following way:

```python
>>> (df.pipe(h)
       .pipe(g, arg1=1)
       .pipe(f, arg2=2, arg3=3)
    )
```

### Using Chaining and Pivot table together

```
(q.query('nationality!= "US"')				
 .sort_values('price', ascending = False)
 .pivot_table(index = 'nationality',
 columns = 'class', 
 aggfunc = {'guest_name':'count', 'price':sum})
)
```

Note that column is not necessary.

## Time Series Data

### Datetime 

First day of the month:

```python
ds.values.astype('datetime64[M]')
```
[Stackoverflow Answer](https://stackoverflow.com/questions/45304531/extracting-the-first-day-of-month-of-a-datetime-type-column-in-pandas)

### Date Format

| format | Value |
-------- | --------
`%d`| two digits date
`%m`| two digits month
`%Y`| four digit year

 

## Data Tricks

### Change Pandas display format to dollar

Below code will change the global setting and will turn all floats into '$' format.

```python
import pandas as pd
pd.options.display.float_format = '${:,.2f}'.format
df = pd.DataFrame([123.4567, 234.5678, 345.6789, 456.7890],
                  index=['foo','bar','baz','quux'],
                  columns=['cost'])
print(df)
```

But if you don't want the formatting to be applied globally.

```python
import pandas as pd
df['foo'] = df['cost']
df['cost'] = df['cost'].map('${:,.2f}'.format)
print(df)
```

[StackOverFlow](https://stackoverflow.com/questions/20937538/how-to-display-pandas-dataframe-of-floats-using-a-format-string-for-columns)


### Convert Pandas Series to DF


```python
import pandas as pd
pd.Series(loan_type_dict).to_frame().reset_index().rename(columns= {'index':'loan_typ', 0: 'loan_cat'})
```
[StackOverFlow](https://stackoverflow.com/questions/26097916/python-best-way-to-convert-a-pandas-series-into-a-pandas-dataframe)



### Convert date to quarter

The below method has better performance and greater data consistency.

```python
q["qtr"] = pd.to_datetime(q['trans_dt'].values, format='%Y-%m').astype('period[Q]')
```

## PySpark

### Output

PySpark returns 

#### Save PySpark Dataframe as CSV

```python
(pyspark_df
 .coalesce(1)
 .write.format("com.databricks.spark.csv")
 .option("header", "true")
 .save("df_output.csv"))
```




## Debug Log

#### local variable (?) referenced before assignment

```python
import numpy as np

def assign_name(p):
    if 'abc' in p.lower():
        result0 = 'ABC'
    elif 'def' in p.lower():
        result0 = 'DEF'
    return result0

data_file_sub['diaster_impact'].map(diaster_name)

# Error
UnboundLocalError: local variable 'result0' referenced before assignment
```

The issue is caused by not specifying `else` statement.

#### TypeError: argument of type 'builtin_function_or_method' is not iterable

```
def assign_name(p):
    if 'abc' in p.lower:
        result0 = 'ABC'
    elif 'def' in p.lower:
        result0 = 'DEF'
    return result0
```

See below explanation:

[TypeError: argument of type 'builtin_function_or_method' is not iterable](https://stackoverflow.com/questions/14414720/python-typeerror-argument-of-type-builtin-function-or-method-is-not-iterable)

## Additional Resources

[How to Think Like a Computer Scientist with Python 3](http://openbookproject.net/thinkcs/python/english3e/index.html)

# Appendix

### Windows Python Setup Basics

[A little Python and pip basics](https://stackoverflow.com/questions/29817447/how-to-run-pip-commands-from-cmd)

#### Set environment variable
Below line add Python and scripts folders (which contains `pip`) to the environment path.

`$env:path="$env:Path;C:\Python27;C:\Python27\Scripts;C:\Users\rgu10\Downloads\chromedriver_win32;C:\\Python27;C:\\Python27\\Scripts"`

After adding to the path, `python` and `pip` can be invoke from command line directly.

#### Install pip  and setup tools
`python -m pip install -U pip setuptools`

#### Use Virtual Environment
`pip install virtualenv`

#### Basic Packages
```python
pip install pandas # pandas includes numpy
```

#### List all local Python packages

```ptyhon
import pip
sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()])
```

### IPython/Jupyter

#### Display all outputs in IPython/Jupyter

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

[How to display full output in Jupyter, not only last result?](https://stackoverflow.com/questions/36786722/how-to-display-full-output-in-jupyter-not-only-last-result)

#### Display all columns in IPython/Jupyter

```python
pandas.set_option('display.max_columns', None)
```

[Output data from all columns in a dataframe in pandas](https://stackoverflow.com/questions/11361985/output-data-from-all-columns-in-a-dataframe-in-pandas)

#### Working with Bokeh

In order to use Bokeh in Jupyter, we need to use the `output_notebook()` function.

```python
from bokeh.io import output_notebook
output_notebook()
```

### Additional resources

[Working with Data Using pandas and Python 3](https://www.digitalocean.com/community/tutorials/working-with-data-using-pandas-and-python-3)

### Reload package

Python 2.7
```python
# built-in function
reload(package)
```

Python 3.0 to 3.3
```python
import imp
imp.reload(package)
```

Python 3.4+
```python
import importlib
importlib.reload(package)
```

https://stackoverflow.com/questions/32234156/how-to-unimport-a-python-module-which-is-already-imported

### Check python version

```python
import sys
print (sys.version)
```

https://stackoverflow.com/questions/1093322/how-do-i-check-what-version-of-python-is-running-my-script

### Building minimal python package

http://python-packaging.readthedocs.io/en/latest/minimal.html

### Create Python package

The basic python package structure is as follow:

```
pyutil\
	__init__.py # empty
	testfunc\
		__init__.py # empty
		functions.py
			|- fk_to_dt()
```

#### Sure fire way
`pyutil\__init__.py`:  empty
`pyutil\testfunc\__init__.py`:  empty

```python
>>> from pyutil.testfunc import functions as uf
>>> uf.fk_to_dt()
```

#### Making things easier
`pyutil\__init__.py`:  empty
`pyutil\testfunc\__init__.py`:  add `__all__ = ['functions']`

```python
>>> from pyutil.testfunc import *
>>> functions.fk_to_dt() # still need to refer to the name of the submodule
```

#### Folding code at a higher level

We want to import all the functions at `testfunc` level, so that we don't need to call each function specifically. This can be done by letting the `__init__.py` import all functions in the module when calling the module.

In `pyutil\__init__.py`:  empty
In `pyutil\testfunc\__init__.py`:  
```python
__all__ = ['functions']
from .functions import * 
```
The `.` in `.functions` as the script to source sibling files.

```python
>>> from pyutil import testfunc as uf
>>> uf.fk_to_dt()
```

We can go even a step further on this.

```python
# pyutil\__init__.py
__all__ = ['testfunc']
from .testfunc import * 

# pyutil\testfunc\__init__.py
__all__ = ['functions']
from .functions import * 
```

This way we can skip calling `testfunc` explicitly while keeping a clean file path.

```python
>>> from pyutil import functions as uf
>>> uf.fk_to_dt()
```

However, the below will not work. 

```python
>>> from pyutil import *
>>> functions.fk_to_dt() # fk_to_dt() not found
>>> fk_to_dt() # not found 
```

Below also will not work.

```python
>>> import pyutil as uf
>>> uf.fk_to_dt() # uf.fk_to_dt() not found
```

#### The laziest way

We can explicitly call sub-modules in the init script so that all code will fold at the highest level.
 
```python
# pyutil\__init__.py
__all__ = ['testfunc']
from .testfunc.functions import * 

# pyutil\testfunc\__init__.py
__all__ = ['functions']
from .functions import * # this is not even necessary
```

Now this will work.

```python
>>> import pyutil as uf
>>> uf.fk_to_dt()
```

This will also work.

```python
>>> from pyutil import fk_to_dt
>>> fk_to_dt()
```
