#! /bin/bash
function print () { echo "$(date): $@"; }

print "Started untaring to TMPDIR"
tar -xf $WORK/leggedgym.tar -C ${TMPDIR}
print "Sucessfully untared to TMPDIR. Starting singularity exec."
singularity exec ${custom_flags} ${TMPDIR}/leggedgym.sif bash -c "${run_cmd}"
