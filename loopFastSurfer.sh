DATA_DIR="/home/jovyan/shared/data/OASIS-2/OAS2_RAW_PART1/"
THIS_SUBJ="OAS2_0002_MR1"
SETUP_DIR=$/home/jovyan/FastSurfer
THIS_FILE="/home/jovyan/shared/data/OASIS-2/OAS2_RAW_PART1/" $THIS_SUBJ "/RAW/mpr-1.nifti.hdr"
cd /home/jovyan/FastSurfer
./run_fastsurfer.sh --t1 $THIS_FILE \
                                     --sd "{SETUP_DIR}fastsurfer_seg" \
                                     --sid $THIS_SUBJ \
                                     --seg_only --py python3 \
                                     --no_biasfield --no_cereb --no_hypothal \
                                     --allow_root
