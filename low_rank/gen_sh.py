# a =" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234"  

# b = " main_swin_calkl_s2.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume pretrainmodel/swin_large_patch4_window7_224_22kto1k.pth --batch-size 64  --layer-i 3 --eval  --block-i "
# d = [" --fc-i 1 --dim-i 32", " --fc-i 1 --dim-i 64", " --fc-i 1 --dim-i 96", " --fc-i 1 --dim-i 128", 
#     " --fc-i 1 --dim-i 160", " --fc-i 1 --dim-i 192", " --fc-i 1 --dim-i 224", " --fc-i 1 --dim-i 256", " --fc-i 1 --dim-i 288", 
#     " --fc-i 1 --dim-i 320", " --fc-i 1 --dim-i 352", " --fc-i 1 --dim-i 384",  
#     " --fc-i 1 --dim-i 416", " --fc-i 1 --dim-i 448", " --fc-i 1 --dim-i 480", " --fc-i 1 --dim-i 512", " --fc-i 1 --dim-i 544", 
#     " --fc-i 1 --dim-i 576", " --fc-i 1 --dim-i 608", 
#     " --fc-i 1 --dim-i 640", " --fc-i 1 --dim-i 672", " --fc-i 1 --dim-i 704", " --fc-i 1 --dim-i 736", " --fc-i 1 --dim-i 768", 
#     " --fc-i 1 --dim-i 800", " --fc-i 1 --dim-i 832",   " --fc-i 1 --dim-i 864",   " --fc-i 1 --dim-i 896",   " --fc-i 1 --dim-i 928",   " --fc-i 1 --dim-i 960",   " --fc-i 1 --dim-i 992",   " --fc-i 1 --dim-i 1024",   " --fc-i 1 --dim-i 1056",   " --fc-i 1 --dim-i 1088",   " --fc-i 1 --dim-i 1120",   " --fc-i 1 --dim-i 1152",   " --fc-i 1 --dim-i 1184",   " --fc-i 1 --dim-i 1216",  
#     " --fc-i 2 --dim-i 32", " --fc-i 2 --dim-i 64", " --fc-i 2 --dim-i 96", " --fc-i 2 --dim-i 128", 
#     " --fc-i 2 --dim-i 160", " --fc-i 2 --dim-i 192", " --fc-i 2 --dim-i 224", " --fc-i 2 --dim-i 256", " --fc-i 2 --dim-i 288", 
#     " --fc-i 2 --dim-i 320", " --fc-i 2 --dim-i 352", " --fc-i 2 --dim-i 384", 
#     " --fc-i 2 --dim-i 416", " --fc-i 2 --dim-i 448", " --fc-i 2 --dim-i 480", " --fc-i 2 --dim-i 512", " --fc-i 2 --dim-i 544", 
#     " --fc-i 2 --dim-i 576", " --fc-i 2 --dim-i 608", 
#     " --fc-i 2 --dim-i 640", " --fc-i 2 --dim-i 672", " --fc-i 2 --dim-i 704", " --fc-i 2 --dim-i 736", " --fc-i 2 --dim-i 768", 
#     " --fc-i 2 --dim-i 800", " --fc-i 2 --dim-i 832",   " --fc-i 2 --dim-i 864",   " --fc-i 2 --dim-i 896",   " --fc-i 2 --dim-i 928",   " --fc-i 2 --dim-i 960",   " --fc-i 2 --dim-i 992",   " --fc-i 2 --dim-i 1024",   " --fc-i 2 --dim-i 1056",   " --fc-i 2 --dim-i 1088",   " --fc-i 2 --dim-i 1120",   " --fc-i 2 --dim-i 1152",   " --fc-i 2 --dim-i 1184",   " --fc-i 2 --dim-i 1216",  
#     " --fc-i 3 --dim-i 32", " --fc-i 3 --dim-i 64", " --fc-i 3 --dim-i 96", " --fc-i 3 --dim-i 128", 
#     " --fc-i 3 --dim-i 160", " --fc-i 3 --dim-i 192", " --fc-i 3 --dim-i 224", " --fc-i 3 --dim-i 256", " --fc-i 3 --dim-i 288",
#     " --fc-i 3 --dim-i 320", " --fc-i 3 --dim-i 352", " --fc-i 3 --dim-i 384", 
#     " --fc-i 3 --dim-i 416", " --fc-i 3 --dim-i 448", " --fc-i 3 --dim-i 480", " --fc-i 3 --dim-i 512", " --fc-i 3 --dim-i 544", 
#     " --fc-i 3 --dim-i 576",  " --fc-i 3 --dim-i 608", 
#     " --fc-i 3 --dim-i 640", " --fc-i 3 --dim-i 672", " --fc-i 3 --dim-i 704", " --fc-i 3 --dim-i 736", " --fc-i 3 --dim-i 768", 
#     " --fc-i 3 --dim-i 800", " --fc-i 3 --dim-i 832",   " --fc-i 3 --dim-i 864",   " --fc-i 3 --dim-i 896",   " --fc-i 3 --dim-i 928",   " --fc-i 3 --dim-i 960",   " --fc-i 3 --dim-i 992",   " --fc-i 3 --dim-i 1024",   " --fc-i 3 --dim-i 1056",   " --fc-i 3 --dim-i 1088",   " --fc-i 3 --dim-i 1120",   " --fc-i 3 --dim-i 1152",   " --fc-i 3 --dim-i 1184",   " --fc-i 3 --dim-i 1216",  
#     ]

# mul = 0
# cuda_ind = 5
# for c in range(2 * mul, 2 *(mul + 1) ):
#     for i in d:
#         print("CUDA_VISIBLE_DEVICES=" + str(cuda_ind) + a + str(cuda_ind) +  b + str(c) + i)


# Swin B
# a =" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234"  

# b = " main_swin_calkl_s2.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume pretrainmodel/swin_base_patch4_window7_224.pth --batch-size 64  --layer-i 3 --eval  --block-i "
# d = [" --fc-i 1 --dim-i 32", " --fc-i 1 --dim-i 64", " --fc-i 1 --dim-i 96",    
#     " --fc-i 1 --dim-i 128", " --fc-i 1 --dim-i 160", " --fc-i 1 --dim-i 192", 
#     " --fc-i 1 --dim-i 224", " --fc-i 1 --dim-i 256", " --fc-i 1 --dim-i 288",  
#     " --fc-i 1 --dim-i 320", " --fc-i 1 --dim-i 352", " --fc-i 1 --dim-i 384",  
#     " --fc-i 1 --dim-i 416", " --fc-i 1 --dim-i 448", " --fc-i 1 --dim-i 480", " --fc-i 1 --dim-i 512", " --fc-i 1 --dim-i 544", " --fc-i 1 --dim-i 576", 
#     " --fc-i 1 --dim-i 608", " --fc-i 1 --dim-i 640", " --fc-i 1 --dim-i 672", " --fc-i 1 --dim-i 704", " --fc-i 1 --dim-i 736", " --fc-i 1 --dim-i 768", 
#     " --fc-i 1 --dim-i 800", 
#     " --fc-i 2 --dim-i 32", " --fc-i 2 --dim-i 64", " --fc-i 2 --dim-i 96",    
#     " --fc-i 2 --dim-i 128", " --fc-i 2 --dim-i 160", " --fc-i 2 --dim-i 192", 
#     " --fc-i 2 --dim-i 224", " --fc-i 2 --dim-i 256", " --fc-i 2 --dim-i 288", 
#     " --fc-i 2 --dim-i 320", " --fc-i 2 --dim-i 352", " --fc-i 2 --dim-i 384", 
#     " --fc-i 2 --dim-i 416", " --fc-i 2 --dim-i 448", " --fc-i 2 --dim-i 480", " --fc-i 2 --dim-i 512", " --fc-i 2 --dim-i 544", " --fc-i 2 --dim-i 576", 
#     " --fc-i 2 --dim-i 608", " --fc-i 2 --dim-i 640", " --fc-i 2 --dim-i 672", " --fc-i 2 --dim-i 704", " --fc-i 2 --dim-i 736", " --fc-i 2 --dim-i 768", 
#     " --fc-i 2 --dim-i 800", 
#     " --fc-i 3 --dim-i 32", " --fc-i 3 --dim-i 64", " --fc-i 3 --dim-i 96",    
#     " --fc-i 3 --dim-i 128", " --fc-i 3 --dim-i 160", " --fc-i 3 --dim-i 192", 
#     " --fc-i 3 --dim-i 224", " --fc-i 3 --dim-i 256", " --fc-i 3 --dim-i 288", 
#     " --fc-i 3 --dim-i 320", " --fc-i 3 --dim-i 352", " --fc-i 3 --dim-i 384", 
#     " --fc-i 3 --dim-i 416", " --fc-i 3 --dim-i 448", " --fc-i 3 --dim-i 480", " --fc-i 3 --dim-i 512", " --fc-i 3 --dim-i 544", " --fc-i 3 --dim-i 576", 
#     " --fc-i 3 --dim-i 608", " --fc-i 3 --dim-i 640", " --fc-i 3 --dim-i 672", " --fc-i 3 --dim-i 704", " --fc-i 3 --dim-i 736", " --fc-i 3 --dim-i 768", 
#     " --fc-i 3 --dim-i 800"]

# mul = 0
# cuda_ind = 5
# for c in range(2 * mul, 2 *(mul + 1) ):
#     for i in d:
#         print("CUDA_VISIBLE_DEVICES=" + str(cuda_ind) + a + str(cuda_ind) +  b + str(c) + i)

# Swin S
# mul = 2
# cuda_ind = 7

# a =" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234"  

# b = " main_swin_calkl_s2.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume pretrainmodel/swin_small_patch4_window7_224.pth --batch-size 64  --layer-i 2 --eval  --block-i "
# d = [" --fc-i 1 --dim-i 32", " --fc-i 1 --dim-i 64", 
#     " --fc-i 1 --dim-i 96", " --fc-i 1 --dim-i 128", 
#     " --fc-i 1 --dim-i 160", " --fc-i 1 --dim-i 192", 
#     " --fc-i 1 --dim-i 224", " --fc-i 1 --dim-i 256", " --fc-i 1 --dim-i 288",  
#     # " --fc-i 1 --dim-i 320", " --fc-i 1 --dim-i 352", " --fc-i 1 --dim-i 384",  
#     # " --fc-i 1 --dim-i 416", " --fc-i 1 --dim-i 448", " --fc-i 1 --dim-i 480", " --fc-i 1 --dim-i 512", " --fc-i 1 --dim-i 544", " --fc-i 1 --dim-i 576", 
#     # " --fc-i 1 --dim-i 608", 
#     # " --fc-i 1 --dim-i 640", " --fc-i 1 --dim-i 672", " --fc-i 1 --dim-i 704", " --fc-i 1 --dim-i 736", " --fc-i 1 --dim-i 768", 
#     # " --fc-i 1 --dim-i 800", 
#     " --fc-i 2 --dim-i 32", " --fc-i 2 --dim-i 64", 
#     " --fc-i 2 --dim-i 96", " --fc-i 2 --dim-i 128", 
#     " --fc-i 2 --dim-i 160", " --fc-i 2 --dim-i 192", 
#     " --fc-i 2 --dim-i 224", " --fc-i 2 --dim-i 256", " --fc-i 2 --dim-i 288", 
#     # " --fc-i 2 --dim-i 320", " --fc-i 2 --dim-i 352", " --fc-i 2 --dim-i 384", 
#     # " --fc-i 2 --dim-i 416", " --fc-i 2 --dim-i 448", " --fc-i 2 --dim-i 480", " --fc-i 2 --dim-i 512", " --fc-i 2 --dim-i 544", " --fc-i 2 --dim-i 576", 
#     # " --fc-i 2 --dim-i 608", 
#     # " --fc-i 2 --dim-i 640", " --fc-i 2 --dim-i 672", " --fc-i 2 --dim-i 704", " --fc-i 2 --dim-i 736", " --fc-i 2 --dim-i 768", 
#     # " --fc-i 2 --dim-i 800", 
#     " --fc-i 3 --dim-i 32", " --fc-i 3 --dim-i 64", 
#     " --fc-i 3 --dim-i 96", " --fc-i 3 --dim-i 128", 
#     " --fc-i 3 --dim-i 160", " --fc-i 3 --dim-i 192", 
#     " --fc-i 3 --dim-i 224", " --fc-i 3 --dim-i 256", " --fc-i 3 --dim-i 288", 
#     # " --fc-i 3 --dim-i 320", " --fc-i 3 --dim-i 352", " --fc-i 3 --dim-i 384", 
#     # " --fc-i 3 --dim-i 416", " --fc-i 3 --dim-i 448", " --fc-i 3 --dim-i 480", " --fc-i 3 --dim-i 512", " --fc-i 3 --dim-i 544", " --fc-i 3 --dim-i 576", 
#     # " --fc-i 3 --dim-i 608", 
#     # " --fc-i 3 --dim-i 640", " --fc-i 3 --dim-i 672", " --fc-i 3 --dim-i 704", " --fc-i 3 --dim-i 736", " --fc-i 3 --dim-i 768", 
#     # " --fc-i 3 --dim-i 800"
#     ]

# for c in range(6 * mul, 6 *(mul + 1) ):
#     for i in d:
#         print("CUDA_VISIBLE_DEVICES=" + str(cuda_ind) + a + str(cuda_ind) +  b + str(c) + i)

# Swin T
mul = 0
cuda_ind = 7

a =" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234"  

b = " main_swin_calkl_s2.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume pretrainmodel/swin_tiny_patch4_window7_224.pth --batch-size 64  --layer-i 3 --eval  --block-i "
d = [" --fc-i 1 --dim-i 32", " --fc-i 1 --dim-i 64", " --fc-i 1 --dim-i 96", 
    " --fc-i 1 --dim-i 128", " --fc-i 1 --dim-i 160", " --fc-i 1 --dim-i 192", 
    " --fc-i 1 --dim-i 224", " --fc-i 1 --dim-i 256", " --fc-i 1 --dim-i 288",  
    " --fc-i 1 --dim-i 320", " --fc-i 1 --dim-i 352", " --fc-i 1 --dim-i 384",  
    " --fc-i 1 --dim-i 416", " --fc-i 1 --dim-i 448", " --fc-i 1 --dim-i 480", 
    " --fc-i 1 --dim-i 512", " --fc-i 1 --dim-i 544", " --fc-i 1 --dim-i 576", 
    " --fc-i 1 --dim-i 608", " --fc-i 1 --dim-i 640", " --fc-i 1 --dim-i 672", 
    " --fc-i 1 --dim-i 704", " --fc-i 1 --dim-i 736", " --fc-i 1 --dim-i 768", 
    " --fc-i 2 --dim-i 32", " --fc-i 2 --dim-i 64", " --fc-i 2 --dim-i 96", 
    " --fc-i 2 --dim-i 128", " --fc-i 2 --dim-i 160", " --fc-i 2 --dim-i 192", 
    " --fc-i 2 --dim-i 224", " --fc-i 2 --dim-i 256", " --fc-i 2 --dim-i 288",  
    " --fc-i 2 --dim-i 320", " --fc-i 2 --dim-i 352", " --fc-i 2 --dim-i 384",  
    " --fc-i 2 --dim-i 416", " --fc-i 2 --dim-i 448", " --fc-i 2 --dim-i 480", 
    " --fc-i 2 --dim-i 512", " --fc-i 2 --dim-i 544", " --fc-i 2 --dim-i 576", 
    " --fc-i 2 --dim-i 608", " --fc-i 2 --dim-i 640", " --fc-i 2 --dim-i 672", 
    " --fc-i 2 --dim-i 704", " --fc-i 2 --dim-i 736", " --fc-i 2 --dim-i 768", 
    " --fc-i 3 --dim-i 32", " --fc-i 3 --dim-i 64", " --fc-i 3 --dim-i 96", 
    " --fc-i 3 --dim-i 128", " --fc-i 3 --dim-i 160", " --fc-i 3 --dim-i 192", 
    " --fc-i 3 --dim-i 224", " --fc-i 3 --dim-i 256", " --fc-i 3 --dim-i 288",  
    " --fc-i 3 --dim-i 320", " --fc-i 3 --dim-i 352", " --fc-i 3 --dim-i 384",  
    " --fc-i 3 --dim-i 416", " --fc-i 3 --dim-i 448", " --fc-i 3 --dim-i 480", 
    " --fc-i 3 --dim-i 512", " --fc-i 3 --dim-i 544", " --fc-i 3 --dim-i 576", 
    " --fc-i 3 --dim-i 608", " --fc-i 3 --dim-i 640", " --fc-i 3 --dim-i 672", 
    " --fc-i 3 --dim-i 704", " --fc-i 3 --dim-i 736", " --fc-i 3 --dim-i 768", 
    ]

for c in range(6 * mul, 6 *(mul + 1) ):
    for i in d:
        print("CUDA_VISIBLE_DEVICES=" + str(cuda_ind) + a + str(cuda_ind) +  b + str(c) + i)


## Deit
# a =" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234"  

# b = " main_deit_calkl_s2.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume pretrainmodel/deit_base_patch16_224-b5f2ef4d.pth --batch-size 64   --eval  --block-i "
# d = [\
#     " --fc-i 1 --dim-i 64", " --fc-i 1 --dim-i 96", " --fc-i 1 --dim-i 128", " --fc-i 1 --dim-i 160", " --fc-i 1 --dim-i 192", " --fc-i 1 --dim-i 224", " --fc-i 1 --dim-i 256", " --fc-i 1 --dim-i 288", " --fc-i 1 --dim-i 320", " --fc-i 1 --dim-i 352", " --fc-i 1 --dim-i 384",  " --fc-i 1 --dim-i 416", " --fc-i 1 --dim-i 448", " --fc-i 1 --dim-i 480", " --fc-i 1 --dim-i 512", " --fc-i 1 --dim-i 544", " --fc-i 1 --dim-i 576", " --fc-i 1 --dim-i 608", " --fc-i 1 --dim-i 640", \
#     " --fc-i 2 --dim-i 64", " --fc-i 2 --dim-i 96", " --fc-i 2 --dim-i 128", " --fc-i 2 --dim-i 160", " --fc-i 2 --dim-i 192", " --fc-i 2 --dim-i 224", " --fc-i 2 --dim-i 256", " --fc-i 2 --dim-i 288", " --fc-i 2 --dim-i 320", " --fc-i 2 --dim-i 352", " --fc-i 2 --dim-i 384",  " --fc-i 2 --dim-i 416", " --fc-i 2 --dim-i 448", " --fc-i 2 --dim-i 480", " --fc-i 2 --dim-i 512", " --fc-i 2 --dim-i 544", " --fc-i 2 --dim-i 576", " --fc-i 2 --dim-i 608", " --fc-i 2 --dim-i 640", \
#     " --fc-i 3 --dim-i 64", " --fc-i 3 --dim-i 96", " --fc-i 3 --dim-i 128", " --fc-i 3 --dim-i 160", " --fc-i 3 --dim-i 192", " --fc-i 3 --dim-i 224", " --fc-i 3 --dim-i 256", " --fc-i 3 --dim-i 288", " --fc-i 3 --dim-i 320", " --fc-i 3 --dim-i 352", " --fc-i 3 --dim-i 384",  " --fc-i 3 --dim-i 416", " --fc-i 3 --dim-i 448", " --fc-i 3 --dim-i 480", " --fc-i 3 --dim-i 512", " --fc-i 3 --dim-i 544", " --fc-i 3 --dim-i 576", " --fc-i 3 --dim-i 608", " --fc-i 3 --dim-i 640", \
#         ]
# mul = 5
# for c in range(2 * mul, 2 *(mul + 1) ):
#     g = c // 2
#     for i in d:
#         print("CUDA_VISIBLE_DEVICES=" + str(g) + a + str(g) +  b + str(c) + i)

# a =" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234"  

# b = " main_deit_calkl_s2.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume pretrainmodel/deit_small_patch16_224-cd65a155.pth --batch-size 64   --eval  --block-i "
# d = [\
#     " --fc-i 1 --dim-i 64", " --fc-i 1 --dim-i 96", " --fc-i 1 --dim-i 128", " --fc-i 1 --dim-i 160", " --fc-i 1 --dim-i 192", " --fc-i 1 --dim-i 224", " --fc-i 1 --dim-i 256", " --fc-i 1 --dim-i 288", 
#     # " --fc-i 1 --dim-i 320", " --fc-i 1 --dim-i 352", " --fc-i 1 --dim-i 384",  " --fc-i 1 --dim-i 416", " --fc-i 1 --dim-i 448", " --fc-i 1 --dim-i 480", " --fc-i 1 --dim-i 512", " --fc-i 1 --dim-i 544", " --fc-i 1 --dim-i 576", " --fc-i 1 --dim-i 608", " --fc-i 1 --dim-i 640", \
#     " --fc-i 2 --dim-i 64", " --fc-i 2 --dim-i 96", " --fc-i 2 --dim-i 128", " --fc-i 2 --dim-i 160", " --fc-i 2 --dim-i 192", " --fc-i 2 --dim-i 224", " --fc-i 2 --dim-i 256", " --fc-i 2 --dim-i 288", 
#     # " --fc-i 2 --dim-i 320", " --fc-i 2 --dim-i 352", " --fc-i 2 --dim-i 384",  " --fc-i 2 --dim-i 416", " --fc-i 2 --dim-i 448", " --fc-i 2 --dim-i 480", " --fc-i 2 --dim-i 512", " --fc-i 2 --dim-i 544", " --fc-i 2 --dim-i 576", " --fc-i 2 --dim-i 608", " --fc-i 2 --dim-i 640", \
#     " --fc-i 3 --dim-i 64", " --fc-i 3 --dim-i 96", " --fc-i 3 --dim-i 128", " --fc-i 3 --dim-i 160", " --fc-i 3 --dim-i 192", " --fc-i 3 --dim-i 224", " --fc-i 3 --dim-i 256", " --fc-i 3 --dim-i 288", 
#     # " --fc-i 3 --dim-i 320", " --fc-i 3 --dim-i 352", " --fc-i 3 --dim-i 384",  " --fc-i 3 --dim-i 416", " --fc-i 3 --dim-i 448", " --fc-i 3 --dim-i 480", " --fc-i 3 --dim-i 512", " --fc-i 3 --dim-i 544", " --fc-i 3 --dim-i 576", " --fc-i 3 --dim-i 608", " --fc-i 3 --dim-i 640", \
#         ]

# mul = 5
# for c in range(2 * mul, 2 *(mul + 1) ):
#     g = c // 2
#     for i in d:
#         print("CUDA_VISIBLE_DEVICES=" + str(g) + a + str(g) +  b + str(c) + i)

# a =" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234"  

# b = " main_deit_calkl_s2.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /mnt/ramdisk2/ImageNet/ --resume pretrainmodel/deit_tiny_patch16_224-a1311bcf.pth --batch-size 64   --eval  --block-i "
# d = [\
#     " --fc-i 1 --dim-i 32", " --fc-i 1 --dim-i 64", " --fc-i 1 --dim-i 96", " --fc-i 1 --dim-i 128", " --fc-i 1 --dim-i 160", " --fc-i 1 --dim-i 192", 
#     # " --fc-i 1 --dim-i 224", " --fc-i 1 --dim-i 256", " --fc-i 1 --dim-i 288", 
#     # " --fc-i 1 --dim-i 320", " --fc-i 1 --dim-i 352", " --fc-i 1 --dim-i 384",  " --fc-i 1 --dim-i 416", " --fc-i 1 --dim-i 448", " --fc-i 1 --dim-i 480", " --fc-i 1 --dim-i 512", " --fc-i 1 --dim-i 544", " --fc-i 1 --dim-i 576", " --fc-i 1 --dim-i 608", " --fc-i 1 --dim-i 640", \
#     " --fc-i 2 --dim-i 32", " --fc-i 2 --dim-i 64", " --fc-i 2 --dim-i 96", " --fc-i 2 --dim-i 128", " --fc-i 2 --dim-i 160", " --fc-i 2 --dim-i 192", 
#     # " --fc-i 2 --dim-i 224", " --fc-i 2 --dim-i 256", " --fc-i 2 --dim-i 288", 
#     # " --fc-i 2 --dim-i 320", " --fc-i 2 --dim-i 352", " --fc-i 2 --dim-i 384",  " --fc-i 2 --dim-i 416", " --fc-i 2 --dim-i 448", " --fc-i 2 --dim-i 480", " --fc-i 2 --dim-i 512", " --fc-i 2 --dim-i 544", " --fc-i 2 --dim-i 576", " --fc-i 2 --dim-i 608", " --fc-i 2 --dim-i 640", \
#     " --fc-i 3 --dim-i 32", " --fc-i 3 --dim-i 64", " --fc-i 3 --dim-i 96", " --fc-i 3 --dim-i 128", " --fc-i 3 --dim-i 160", " --fc-i 3 --dim-i 192", 
#     # " --fc-i 3 --dim-i 224", " --fc-i 3 --dim-i 256", " --fc-i 3 --dim-i 288", 
#     # " --fc-i 3 --dim-i 320", " --fc-i 3 --dim-i 352", " --fc-i 3 --dim-i 384",  " --fc-i 3 --dim-i 416", " --fc-i 3 --dim-i 448", " --fc-i 3 --dim-i 480", " --fc-i 3 --dim-i 512", " --fc-i 3 --dim-i 544", " --fc-i 3 --dim-i 576", " --fc-i 3 --dim-i 608", " --fc-i 3 --dim-i 640", \
#         ]

# mul = 0
# for c in range(3 * mul, 3 *(mul + 1) ):
#     g = c // 3
#     for i in d:
#         print("CUDA_VISIBLE_DEVICES=" + str(g+4) + a + str(g) +  b + str(c) + i)