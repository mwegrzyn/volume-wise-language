import seaborn as sns
sns.set_style('ticks')
sns.set()

from .data import make_dataset_z_orig
from .data import make_dataset_z_diff
from .features import build_features
from .models import predict_model
from .visualization import visualize_time
from .visualization import visualize_counts
from .visualization import visualize_log
from .visualization import visualize_all

def make_predict(pFolder,pName):
    """Run the whole volume-wise analysis

    This function collects all the necessary modules and runs them
    in order, to replicate the results of the manuscript for a single patient
    using previously trained models.
    The functions which are being used have been imported from jupyter notebooks
    using the makeSrc.sh script in the helper/ folder.

    Parameters
    ----------
    pFolder : string
        directory where the data are stored
    pName : string
        in the pFolder, there must be a file wr<pName>.nii.gz (realigned
	and normalized but not smoothed) and a file rp_<pName>.txt
	(could be any kind of confounds, but currently only supporting
	SPM realignment parameters).
        Accordingly, this variable defines what the <pName> string
        of that file is

    Returns
    -------
    
    nothing, instead writes data to pFolder
    """

    make_dataset_z_orig.make_p(pFolder,pName) # 02-mw-make-z-maps.ipynb
    make_dataset_z_diff.make_p(pFolder,pName) # 03-mw-make-difference-ims.ipynb

    build_features.make_p(pFolder,pName)      # 09-mw-correlations-with-template.ipynb

    predict_model.make_p(pFolder,pName)       # 10-mw-train-test-classifier
    
    visualize_time.make_p(pFolder,pName)      # 12-mw-make-correlation-plots-time.ipynb
    visualize_counts.make_p(pFolder,pName)    # 14-mw-prediction-space.ipynb
    visualize_log.make_p(pFolder,pName)       # 15-mw-visualize-logistic-regression.ipynb
    visualize_all.make_p(pFolder,pName)       # 16-mw-individual-patients-plot.ipynb

    return

