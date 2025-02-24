#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "vgpu.h"

KERNEL(init, (int *A, int width, int height))
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        int idx = row * width + col;
        A[idx] = idx + 1;
    }
END_KERNEL

#define RADIUS 1

KERNEL(blur, (int *A, int *B, int width, int height))
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        int value = 0;
        int num_values = 0;
        for (int i = -RADIUS; i < RADIUS + 1; i++) {
            for (int j = -RADIUS; j < RADIUS + 1; j++) {
                int row_i = row + i;
                int col_j = col + j;
                if (0 <= row_i && row_i < height && 0 <= col_j && col_j < width) {
                    value += A[row_i * width + col_j];
                    num_values++;
                }
            }
        }
        B[row * width + col] = value / num_values;
    }
END_KERNEL

void print(int *A, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            printf("%5d", A[row * width + col]);
        }
        printf("\n");
    }
}

int main(void) {
    int width = 20;
    int height = 45;
    int *A = (int *)malloc(width * height * sizeof(int));
    int *B = (int *)malloc(width * height * sizeof(int));
    assert(A && B);

    int *d_A, *d_B;
    CHECK(cudaMalloc((void **)&d_A, width * height * sizeof(int)));
    CHECK(cudaMalloc((void **)&d_B, width * height * sizeof(int)));

    dim3 thread_blocks = dim3(ceil_div(width, 2), ceil_div(height, 4));
    dim3 threads_per_block = dim3(2, 4);
    init(/* <<< */ thread_blocks, threads_per_block /* >>> */, d_A, width, height);

    thread_blocks = dim3(ceil_div(width, 3), ceil_div(height, 2));
    threads_per_block = dim3(3, 2);
    blur(/* <<< */ thread_blocks, threads_per_block /* >>> */, d_A, d_B, width, height);

    CHECK(cudaMemcpy(A, d_A, width * height * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(B, d_B, width * height * sizeof(int), cudaMemcpyDeviceToHost));

    print(A, width, height);
    print(B, width, height);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));

    free(A);
    free(B);

    return 0;
}

// CHECK:    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20
// CHECK:   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40
// CHECK:   41   42   43   44   45   46   47   48   49   50   51   52   53   54   55   56   57   58   59   60
// CHECK:   61   62   63   64   65   66   67   68   69   70   71   72   73   74   75   76   77   78   79   80
// CHECK:   81   82   83   84   85   86   87   88   89   90   91   92   93   94   95   96   97   98   99  100
// CHECK:  101  102  103  104  105  106  107  108  109  110  111  112  113  114  115  116  117  118  119  120
// CHECK:  121  122  123  124  125  126  127  128  129  130  131  132  133  134  135  136  137  138  139  140
// CHECK:  141  142  143  144  145  146  147  148  149  150  151  152  153  154  155  156  157  158  159  160
// CHECK:  161  162  163  164  165  166  167  168  169  170  171  172  173  174  175  176  177  178  179  180
// CHECK:  181  182  183  184  185  186  187  188  189  190  191  192  193  194  195  196  197  198  199  200
// CHECK:  201  202  203  204  205  206  207  208  209  210  211  212  213  214  215  216  217  218  219  220
// CHECK:  221  222  223  224  225  226  227  228  229  230  231  232  233  234  235  236  237  238  239  240
// CHECK:  241  242  243  244  245  246  247  248  249  250  251  252  253  254  255  256  257  258  259  260
// CHECK:  261  262  263  264  265  266  267  268  269  270  271  272  273  274  275  276  277  278  279  280
// CHECK:  281  282  283  284  285  286  287  288  289  290  291  292  293  294  295  296  297  298  299  300
// CHECK:  301  302  303  304  305  306  307  308  309  310  311  312  313  314  315  316  317  318  319  320
// CHECK:  321  322  323  324  325  326  327  328  329  330  331  332  333  334  335  336  337  338  339  340
// CHECK:  341  342  343  344  345  346  347  348  349  350  351  352  353  354  355  356  357  358  359  360
// CHECK:  361  362  363  364  365  366  367  368  369  370  371  372  373  374  375  376  377  378  379  380
// CHECK:  381  382  383  384  385  386  387  388  389  390  391  392  393  394  395  396  397  398  399  400
// CHECK:  401  402  403  404  405  406  407  408  409  410  411  412  413  414  415  416  417  418  419  420
// CHECK:  421  422  423  424  425  426  427  428  429  430  431  432  433  434  435  436  437  438  439  440
// CHECK:  441  442  443  444  445  446  447  448  449  450  451  452  453  454  455  456  457  458  459  460
// CHECK:  461  462  463  464  465  466  467  468  469  470  471  472  473  474  475  476  477  478  479  480
// CHECK:  481  482  483  484  485  486  487  488  489  490  491  492  493  494  495  496  497  498  499  500
// CHECK:  501  502  503  504  505  506  507  508  509  510  511  512  513  514  515  516  517  518  519  520
// CHECK:  521  522  523  524  525  526  527  528  529  530  531  532  533  534  535  536  537  538  539  540
// CHECK:  541  542  543  544  545  546  547  548  549  550  551  552  553  554  555  556  557  558  559  560
// CHECK:  561  562  563  564  565  566  567  568  569  570  571  572  573  574  575  576  577  578  579  580
// CHECK:  581  582  583  584  585  586  587  588  589  590  591  592  593  594  595  596  597  598  599  600
// CHECK:  601  602  603  604  605  606  607  608  609  610  611  612  613  614  615  616  617  618  619  620
// CHECK:  621  622  623  624  625  626  627  628  629  630  631  632  633  634  635  636  637  638  639  640
// CHECK:  641  642  643  644  645  646  647  648  649  650  651  652  653  654  655  656  657  658  659  660
// CHECK:  661  662  663  664  665  666  667  668  669  670  671  672  673  674  675  676  677  678  679  680
// CHECK:  681  682  683  684  685  686  687  688  689  690  691  692  693  694  695  696  697  698  699  700
// CHECK:  701  702  703  704  705  706  707  708  709  710  711  712  713  714  715  716  717  718  719  720
// CHECK:  721  722  723  724  725  726  727  728  729  730  731  732  733  734  735  736  737  738  739  740
// CHECK:  741  742  743  744  745  746  747  748  749  750  751  752  753  754  755  756  757  758  759  760
// CHECK:  761  762  763  764  765  766  767  768  769  770  771  772  773  774  775  776  777  778  779  780
// CHECK:  781  782  783  784  785  786  787  788  789  790  791  792  793  794  795  796  797  798  799  800
// CHECK:  801  802  803  804  805  806  807  808  809  810  811  812  813  814  815  816  817  818  819  820
// CHECK:  821  822  823  824  825  826  827  828  829  830  831  832  833  834  835  836  837  838  839  840
// CHECK:  841  842  843  844  845  846  847  848  849  850  851  852  853  854  855  856  857  858  859  860
// CHECK:  861  862  863  864  865  866  867  868  869  870  871  872  873  874  875  876  877  878  879  880
// CHECK:  881  882  883  884  885  886  887  888  889  890  891  892  893  894  895  896  897  898  899  900

// CHECK:   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   29
// CHECK:   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   39
// CHECK:   41   42   43   44   45   46   47   48   49   50   51   52   53   54   55   56   57   58   59   59
// CHECK:   61   62   63   64   65   66   67   68   69   70   71   72   73   74   75   76   77   78   79   79
// CHECK:   81   82   83   84   85   86   87   88   89   90   91   92   93   94   95   96   97   98   99   99
// CHECK:  101  102  103  104  105  106  107  108  109  110  111  112  113  114  115  116  117  118  119  119
// CHECK:  121  122  123  124  125  126  127  128  129  130  131  132  133  134  135  136  137  138  139  139
// CHECK:  141  142  143  144  145  146  147  148  149  150  151  152  153  154  155  156  157  158  159  159
// CHECK:  161  162  163  164  165  166  167  168  169  170  171  172  173  174  175  176  177  178  179  179
// CHECK:  181  182  183  184  185  186  187  188  189  190  191  192  193  194  195  196  197  198  199  199
// CHECK:  201  202  203  204  205  206  207  208  209  210  211  212  213  214  215  216  217  218  219  219
// CHECK:  221  222  223  224  225  226  227  228  229  230  231  232  233  234  235  236  237  238  239  239
// CHECK:  241  242  243  244  245  246  247  248  249  250  251  252  253  254  255  256  257  258  259  259
// CHECK:  261  262  263  264  265  266  267  268  269  270  271  272  273  274  275  276  277  278  279  279
// CHECK:  281  282  283  284  285  286  287  288  289  290  291  292  293  294  295  296  297  298  299  299
// CHECK:  301  302  303  304  305  306  307  308  309  310  311  312  313  314  315  316  317  318  319  319
// CHECK:  321  322  323  324  325  326  327  328  329  330  331  332  333  334  335  336  337  338  339  339
// CHECK:  341  342  343  344  345  346  347  348  349  350  351  352  353  354  355  356  357  358  359  359
// CHECK:  361  362  363  364  365  366  367  368  369  370  371  372  373  374  375  376  377  378  379  379
// CHECK:  381  382  383  384  385  386  387  388  389  390  391  392  393  394  395  396  397  398  399  399
// CHECK:  401  402  403  404  405  406  407  408  409  410  411  412  413  414  415  416  417  418  419  419
// CHECK:  421  422  423  424  425  426  427  428  429  430  431  432  433  434  435  436  437  438  439  439
// CHECK:  441  442  443  444  445  446  447  448  449  450  451  452  453  454  455  456  457  458  459  459
// CHECK:  461  462  463  464  465  466  467  468  469  470  471  472  473  474  475  476  477  478  479  479
// CHECK:  481  482  483  484  485  486  487  488  489  490  491  492  493  494  495  496  497  498  499  499
// CHECK:  501  502  503  504  505  506  507  508  509  510  511  512  513  514  515  516  517  518  519  519
// CHECK:  521  522  523  524  525  526  527  528  529  530  531  532  533  534  535  536  537  538  539  539
// CHECK:  541  542  543  544  545  546  547  548  549  550  551  552  553  554  555  556  557  558  559  559
// CHECK:  561  562  563  564  565  566  567  568  569  570  571  572  573  574  575  576  577  578  579  579
// CHECK:  581  582  583  584  585  586  587  588  589  590  591  592  593  594  595  596  597  598  599  599
// CHECK:  601  602  603  604  605  606  607  608  609  610  611  612  613  614  615  616  617  618  619  619
// CHECK:  621  622  623  624  625  626  627  628  629  630  631  632  633  634  635  636  637  638  639  639
// CHECK:  641  642  643  644  645  646  647  648  649  650  651  652  653  654  655  656  657  658  659  659
// CHECK:  661  662  663  664  665  666  667  668  669  670  671  672  673  674  675  676  677  678  679  679
// CHECK:  681  682  683  684  685  686  687  688  689  690  691  692  693  694  695  696  697  698  699  699
// CHECK:  701  702  703  704  705  706  707  708  709  710  711  712  713  714  715  716  717  718  719  719
// CHECK:  721  722  723  724  725  726  727  728  729  730  731  732  733  734  735  736  737  738  739  739
// CHECK:  741  742  743  744  745  746  747  748  749  750  751  752  753  754  755  756  757  758  759  759
// CHECK:  761  762  763  764  765  766  767  768  769  770  771  772  773  774  775  776  777  778  779  779
// CHECK:  781  782  783  784  785  786  787  788  789  790  791  792  793  794  795  796  797  798  799  799
// CHECK:  801  802  803  804  805  806  807  808  809  810  811  812  813  814  815  816  817  818  819  819
// CHECK:  821  822  823  824  825  826  827  828  829  830  831  832  833  834  835  836  837  838  839  839
// CHECK:  841  842  843  844  845  846  847  848  849  850  851  852  853  854  855  856  857  858  859  859
// CHECK:  861  862  863  864  865  866  867  868  869  870  871  872  873  874  875  876  877  878  879  879
// CHECK:  871  872  873  874  875  876  877  878  879  880  881  882  883  884  885  886  887  888  889  889
