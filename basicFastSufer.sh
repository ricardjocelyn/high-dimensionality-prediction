SETUP_DIR=$/home/jovyan/FastSurfer
THIS_FILE="/home/jovyan/shared/data/OASIS-2/OAS2_RAW_PART1/OAS2_0001_MR1/RAW/mpr-1.nifti.hdr"
cd /home/jovyan/FastSurfer
./run_fastsurfer.sh --t1 $THIS_FILE \
                                     --sd "{SETUP_DIR}fastsurfer_seg" \
                                     --sid Tutorial \
                                     --seg_only --py python3 \
                                     --no_biasfield --no_cereb --no_hypothal \
                                     --allow_root
