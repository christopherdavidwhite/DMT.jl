using Dates

subdate = today()
L = 1025

mkpath("$subdate")

dt = 0.125
cutoffs =  [10.0 ^ (-j) for j = 1:2:13]
@show cutoffs
for cutoff = cutoffs
    for maxdim = ( 2 .^ (4:9) .|> floor .|> Int )
        memory = 3000 + ((( maxdim^2*4*L ) * 8 * 1.5 / 2^20) |> ceil |> Int )
        job_script_fn = "$subdate/T111-$cutoff-$maxdim.sub"
        job_script_string = "#!/bin/bash
#SBATCH -p standard
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 6-0
#SBATCH --mem=$memory

source ~/.bashrc
OPENBLAS_NUM_THREADS=2 julia ~/2014-12-TEBD/scripts/TEBD111_majorana-model-cutoff/TEBD111_majorana-model-cutoff-cluster.jl -L $L --maxdim $maxdim --time 200 --dt $dt --subdate $subdate -U 0.3 --cutoff $cutoff
"  
        println(job_script_string)
        f = open(job_script_fn, "w")
        println(f, job_script_string)
        close(f)
    end
end


