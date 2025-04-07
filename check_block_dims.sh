#!/usr/bin/env bash

set -eu

cat << EOF > check_block_dims.c
#include <assert.h>
#include <stdio.h>
#include "vgpu.h"

KERNEL(check, ()) END_KERNEL

int main(int argc, char *argv[]) {
    assert(argc == 5);

    int grid_x = atoi(argv[1]);
    int grid_y = atoi(argv[2]);
    int grid_z = 1;

    int block_x = atoi(argv[3]);
    int block_y = atoi(argv[4]);
    int block_z = 1;

    dim3 grid = dim3(grid_x, grid_y, grid_z);
    dim3 blocks = dim3(block_x, block_y, block_z);
    check(/* <<< */ grid, blocks /* >>> */);

    return 0;
}
EOF

make -s check_block_dims

echo " Supported block dimensions"
echo "+---------+---------+---------+"
echo "|    x    |    y    |    z    |"
echo "+---------+---------+---------+"

for x in {1..512}; do
    for y in {1..512}; do
        if eval ./check_block_dims 1 1 "$x" "$y" 2> /dev/null; then
            printf "|   %3d   |   %3d   |   %3d   |\n" "$x" "$y" 1
        else
            break
        fi
    done
done

echo "+---------+---------+---------+"

rm check_block_dims{.c,}
