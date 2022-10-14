# aggfly
A more permanent home for my climate data aggregation code

Will add docs and instructions in the next few weeks as I have time. For questions in the meantime, feel free to email me at `dth2133@columbia.edu`.

The `requirements.txt` file provides the basic packages you'll need to run the code. Alternatively, you can use the `environment.yml` file to recreate the conda environment I use to develop the package. Check out the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) if you need help using it.

Once you have all the dependencies, you can install this package by running `pip install -e .` from the root directory of the repository.

Of course, please feel free to suggest changes/improvements/extensions in [Issues](https://github.com/dylanhogan/aggfly/issues) :)



## Set up notes

(for mac users) you can install a new conda environment with the basic requirements using the following commands
```
cd "{path_to_repo}"
conda update -n base -c defaults conda
conda create --name aggfly python=3.8 pip jupyter ipykernel
conda activate aggfly
pip install -r requirements.txt  
pip install -e .
```

