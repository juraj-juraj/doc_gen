# Code gen4

## Introduction

This thesis aims to create a tool, that would create docstring documentation for Python code.

## Docstrings

Docstrings are the way to document code in Python. They are written in the first line of function, class or module. They are written in triple quotes. There are many styles of docstrings, like Google style, Numpy style, Sphinx style, reST etc.

Docstrings are formally specified in [PEP 257](https://www.python.org/dev/peps/pep-0257/). This convention talks about how docstrings should be syntactically written, but it does not specify how they should be written semantically. This is left to the programmer. The docstring of each function can be retrieved via the `__doc__` attribute of the function.

The next chapters will be about different styles of docstrings.

### Google style

Google defines its style of docstrings. It is described in the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). This convention defines for which function is docstring mandatory and for which is optional. It also defines how the docstring should be written. The docstring should contain the following parts:

* Summary of what the particular function or method does. It is not denoted by any keyword.
* `Args`: List each parameter by name along with a description of what it is.
* `Returns`(`Yields` for generators): Describe the type and meaning of any return values.
* `Raises`: List all exceptions that are relevant to the interface followed by a description. This section is optional.
* `Attributes`: List of all public attributes of the class.

#### Example

```python
    """One line summary.

    In depth description of what the function or method does. 
    Usually on multiple lines.

    Args:
        first: First parameter description.
        second: Second parameter description.

    Returns:
        One or more lines describing the return value or values.

    Raises:
        Exception: In case of an error, an exception is raised.
    """
```

### Numpydoc style

Convention for the Numpydoc style is defined in the official [Numpydoc documentation](https://numpydoc.readthedocs.io/en/latest/format.html). This convention is used by libraries like Numpy, SciPy, scikit-learn, etc.
This convention uses [reStructuredText](http://docutils.sourceforge.net/rst.html) syntax and is rendered using [Sphinx](https://www.sphinx-doc.org/).

Numpydoc style consists mainly of the following sections:

* Short summary line. Not denoted by any keyword.
* `Parameters`: List each parameter by name along with a description of what it is.
* `Attributes`: List of all public attributes of the class. This section is valid only for classes.
* `Returns`: Explanation of the return values and their types.
* `Yields`: Describe yielded values and their types. This section is valid for generators.
* `Raises`: Explain which exceptions can be raised under which conditions. This section is optional.
* `See Also`: References to related code. This section is optional.
* `Notes`: Additional information about the code. This section is optional.
* `Examples`: Examples of usage. This section is optional.

#### Example

```python
    """One line summary.

    In depth description of what the function or method does. 
    Usually on multiple lines.

    Parameters
    ----------
    first :
        First parameter description.
    second :
        Second parameter description.

    Returns
    -------
    int
        One or more lines describing the return value or values.

    Raises
    ------
    Exception
        In case of an error, an exception is raised.
    """
```

### Epytext style

Historically inspired by `javadoc` style, this convention is defined in the [Epytext documentation](https://epydoc.sourceforge.net/). This convention is used by libraries like Twisted, etc. This markup language is suited for generating API documentation. Summary, or description, is defined by constructions similar to markdown. After the description part can be fields which describe specific properties of a documented object. For example parameters, return values, exceptions, etc. The definition of fields can be found in the [Epytext documentation](https://epydoc.sourceforge.net/fields.html#fields).
Let's choose a few fields and describe them:

* `@param`: Description of a parameter. Synonyms are `@arg`, `@argument`, `@parameter`.
* `@type`: Type of a parameter. 
* `@return`: Description of a return value. Synonyms are `@returns`.
* `@rtype`: Type of a return value. Synonyms are `@returntype`.
* `@raise`: Description of an exception.

Custom fields can be defined by the user if needed. This makes this documentation style very flexible.

#### Example

```python
    """
    Summary of what the particular function or method does. It is not denoted by any keyword.

    Section 1
    =========
    In depth description of what the function or method does.
    Many sections can be defined, lists, tables, etc.

    @type  first: int
    @param first: Description of parameter first.
    @type  second: float
    @param second: Description of parameter second.
    @rtype:   bool
    @return:  Description of return value.
    """
```

# Processing dataset

Dataset is filtered by the length of docstring and split to different by different docstring styles.
