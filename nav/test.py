import dataset.dataset_cassini_iss
ds = dataset.dataset_cassini_iss.DataSetCassiniISS()
y=ds.yield_image_filenames_index(index_dir='/mnt/rms-holdings/holdings/metadata', volume_raw_dir='/mnt/rms-holdings/holdings/volumes',
                                 choose_random_images=False, vol_start='COISS_2009', vol_end='COISS_2015')
cnt = 0
for img_path in y:
    print(img_path)
    cnt += 1
    if cnt > 10:
        break
