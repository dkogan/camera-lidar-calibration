#!/bin/zsh

# This is a calibration wrapper script. It uses mrcal to compute a calibration
# using different lens models. Common diagnostics are produced


read -r -d '' usage <<EOF || true
Usage: $0 [--force]                          \\
          [--calibrate-options ...]          \\
          [--diff-options ...]               \\
          --models LENSMODEL0,LENSMODEL1,... \\
          PATH_TO_RESULTS                    \\
          IMAGE_GLOB_SET0                    \\
          IMAGE_GLOB_SET1

SYNOPSIS

  $0 \\
    --models LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=18_Ny=14_fov_x_deg=110,LENSMODEL_OPENCV8 \\
    --calibrate-options '--corners-cache corners.vnl --focal 1000' \\
    /tmp/results      \\
    'image*[0-4].png' \\
    'image*[5-9].png'

DESCRIPTION

All the output will be written to PATH_TO_RESULTS. This directory will be
created. If it already exists and if --force then then I simply write into that
directory, possibly overwriting files.

IT IS ASSUMED THAT THE IMAGE FILENAMES MATCH

  xxxxFRAMENUMBER.???

Otherwise the set0/set1 split is hard. The code checks for this, and will barf if
it is violated.

Arguments:

  --models LENSMODEL0,LENSMODEL1,... This argument is required. It is a
  comma-separated list of all the lens models we solve for. Each LENSMODEL must
  be a LENSMODEL_XXXXX model string known to mrcal. If given at least 2 models I
  will compute the difference between the first two. And I will produce residual
  plots for the first one. This is useful to compare a splined model result
  against some other model we would use instead. It is recommended to pass a
  splined model first and the lean model second

  --calibrate-options '--opt0 --opt1 ...'. This argument is not required, but
  you almost certainty want to pass something. Passed directly to
  mrcal-calibrate-cameras

  --diff-options '--opt0 --opt1...': specifies the options to pass to
    mrcal-show-projection-diff. Defaults to '--no-uncertainties --radius 200'

  --force: if PATH_TO_RESULTS exists, then without --force we will throw an
  error, and do nothing. With --force we write into the existing directory,
  possibly overwriting data

EOF

# The parsing code to use getopt comes from
#   /usr/share/doc/util-linux/examples/getopt-example.bash
# I'm not an expert, but this sample makes things work
TEMP=$(getopt -o '' --long 'force,models:,calibrate-options:,diff-options:' -n "$0" -- "$@")
[ $? -ne 0 ] && {
    echo '' > /dev/stderr
    echo $usage > /dev/stderr
    exit 1;
}

eval set -- "$TEMP"
unset TEMP

diff_options=(--radius 200)

# The arguments now appear in a nice canonical order, and they are terminated by
# --
while {true} {
	case "$1" in
		'--force')
			force=1
			shift
			continue
		;;

		'--models')
                        lensmodels=(${=2//,/ })
			shift 2
			continue
		;;

		'--calibrate-options')
                        calibrate_options=(${(z)2})
			shift 2
			continue
		;;

		'--diff-options')
                        diff_options=(${(z)2})
			shift 2
			continue
		;;

		'--')
			shift
			break
		;;

		*)
			echo 'Internal error in argument parsing' >&2
			exit 1
		;;
	esac
}

if (( $#lensmodels == 0 )) {
       echo "--models MUST be given" > /dev/stderr
       echo '' > /dev/stderr
       echo $usage > /dev/stderr
       exit 1
}

if (( $#* != 3 )) {
       echo "Exactly 3 non-option arguments are required. Got $#* instead" > /dev/stderr
       echo '' > /dev/stderr
       echo $usage > /dev/stderr
       exit 1
}


Dout=$1;      shift;
glob_set0=$1; shift;
glob_set1=$1; shift;

Ncameras=1

typeset -A globs
globs[set0]="$glob_set0"
globs[set1]="$glob_set1"

if [[ -e "$Dout" ]] {
    if [[ -z "$force" ]] {
           echo "Output directory '$Dout' exists already. Delete it, or pass --force" > /dev/stderr
           exit 1
    }
} else {
  mkdir -p $Dout
}

set -e

whats=()
for lensmodel ($lensmodels) whats+=(${${lensmodel/LENSMODEL_/}/_*/})

for ilensmodel (`seq $#lensmodels`) {
    lensmodel=$lensmodels[$ilensmodel]
    what=$whats[$ilensmodel]

    echo "======== calibrating $lensmodel"
    echo "=== stereo solves"
    for set01 (set0 set1) {

        cmd=(mrcal-calibrate-cameras         \
                 $calibrate_options          \
                 --lensmodel $lensmodel      \
                 --outdir /tmp               \
                 ${=globs[$set01]})

        echo "Running $cmd"
        $cmd

        for i (`seq 0 $((Ncameras-1))`) {
            mv /tmp/camera-$i.cameramodel $Dout/camera-$i-$set01-$what.cameramodel
        }

        for i (`seq 0 $((Ncameras-1))`) {
            mrcal-show-projection-uncertainty                               \
                --cbmax 1                                                   \
                --title "Uncertainty at infinity of camera$i: $what model"  \
                --unset key \
                --hardcopy $Dout/uncertainty-$i-$set01-$what.png     \
                $Dout/camera-$i-$set01-$what.cameramodel
        }
    }

    echo "===== computing cross-validation"

    for i (`seq 0 $((Ncameras-1))`) {
        mrcal-show-projection-diff                                                      \
            $diff_options \
            --hardcopy $Dout/cross-validation-diff-$i-$what.png                  \
            --title "Cross-validation diff at infinity for camera$i: $what $model" \
            --unset key \
            --cbmax 2                                                                   \
            $Dout/camera-$i-{set0,set1}-$what.cameramodel
    }

    for i (`seq 0 $((Ncameras-1))`) {
        # Either the set0 or set1 solution can be the "good" one. I
        # arbitrarily pick the set1 one
        ln -fs                                \
            camera-$i-set1-$what.cameramodel  \
            $Dout/camera-$i-$what.cameramodel
    }
}

(( $#lensmodels >= 2)) &&
{
  for i (`seq 0 $((Ncameras-1))`) {
      for set01 (set0 set1) {
          mrcal-show-projection-diff                                              \
              $diff_options \
              --hardcopy $Dout/error-$i-$set01-$whats[1]-$whats[2].png                  \
              --title "Error in the $whats[1]-$whats[2] solve at infinity for camera$i $set01" \
              --unset key \
              --cbmax 2                                                           \
              $Dout/camera-$i-$set01-{$whats[1],$whats[2]}.cameramodel
      }
      # Either the set0 or set1 solution can be the "good" one. I
      # arbitrarily pick the set1 one
      ln -fs                                  \
          error-$i-set1-$whats[1]-$whats[2].png   \
          $Dout/error-$i-$whats[1]-$whats[2].png
  }
}

i=0
for iworst (`seq 0 3`) {
  mrcal-show-residuals-board-observation \
    --vectorscale 100 \
    --from-worst     \
    --hardcopy $Dout/residual-board-fromworst$iworst-$i-$whats[1].png \
    $Dout/camera-$i-$whats[1].cameramodel \
    $iworst
}

echo "================="
echo "DONE. All results live in '$Dout'"
echo "================="
