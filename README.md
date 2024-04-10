# aggfly
A more permanent home for my climate data aggregation code

Will add docs and instructions in the next few weeks as I have time. For questions in the meantime, feel free to email me at `dth2133@columbia.edu`.

The `aggfly-dev-environment.yml` file provides the core packages you'll need to run the code. You can use this file to recreate the conda environment I use to develop the package. Check out the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) if you need help using it. The following command (executed from root directory of the repository) should be sufficient for most users:

```
conda env create --file aggfly-dev-environment.yml -n aggfly-dev 
```

Once you have all the dependencies, you can install this package by running `pip install -e .` from the root directory of the repository.

```
conda activate aggfly-dev
pip install -e .
```

Of course, please feel free to suggest changes/improvements/extensions in [Issues](https://github.com/dylanhogan/aggfly/issues) :)
