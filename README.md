# HCP-Aging-Connectome-Analysis

These scripts incorporate diffusion data from the HCP Aging dataset (https://www.humanconnectome.org/study/hcp-lifespan-aging/data-releases). 

Following standard preprocessing approaches employed with mrtix3's standard pipeline we performed probabilistic tractography to obtain connectivity matrix's for each subject using the AAL atlas
## AAL atlas information can be found here (https://www.gin.cnrs.fr/en/tools/aal/)

Following matrix creation, we utilized python and the brain connectivity toolbox for python to extract graph theory based network measures including nodal strength, transitivity, and global efficiency

### -----------------------------------------
## Dependencies
# mrtrix3
# python
# FSL
# Freesurfer
