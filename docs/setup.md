# Setup python enviorment

If you want to run the notebooks on your computer make sure you install the requirements.txt python dependencies. I recommend using virtual environments, you could set up one for this repository like this:


Create the environment
```
conda create --name {YOURENVNAME} python=3.7
```

Activate the environment
```
conda activate {YOURENVNAME}
```

Install requirements dependencies

```
python -m pip install -r requirements.txt
```


Do you want to run this enviornment in your jupyter? You can add it as a kernel :

```
conda install -c anaconda ipykernel
```

```
python -m ipykernel install --user --name={KERNELNAME}
```

The `{KERNELNAME}` should now appear on your jupyter kernel list!