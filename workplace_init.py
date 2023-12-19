import re, os, glob

def process_file_names(search_folder, format, idx_interval):
    """
    function searches 'search_folder' and its subfolders for files with format 'format'
    it returns file links sorted by 'idx' integer value in name and int is within given interval.
    min and max idx are return
    old logic guide. principle is the same (now its more explicit and transparent + info on missing):
    1) get image links
    2) extract integers in name [.\img3509, .\img351, .\img3510, ....] -> [3509, 351, 3510, ...]
    3) filter by index 
    4) sort order by index [.\img3509, .\img351, .\img3510, ...] -> [.\img351, .\img3509, .\img3510, ...]
    """
    
    image_links = glob.glob(search_folder + f"**/*.{format}", recursive=True)
    #imageLinks = [f'folder/{i}.bmp' for i in list(range(0,3)) + list(range(6,8))]  # dummy test
    interval_start, interval_stop = idx_interval

    assert len(image_links)> 0, "No files inside directory" 

    int_from_filename = lambda x: int(re.findall('\d+', os.path.basename(x))[0])                           
    temp_dict           = {i:None for i in range(interval_start,interval_stop)}   # store ordered
    idx_min, idx_max    = interval_stop, interval_start                           # initialize boundaries
    count = 0                                                                   # num of viable
    for image_link in image_links:
        idx = int_from_filename(image_link)                                     
        if interval_start <= idx < interval_stop:                                 # filter useful
            temp_dict[idx] = image_link
            idx_min = min(idx, idx_min)                                         # update boundaries
            idx_max = max(idx, idx_max)
            count += 1
    assert count> 0, "No resulting images" 
    output = [image_links[-1]] * count                                            # predefine output
    if count == interval_stop - interval_start:                                   # check if interval is filled
        output = list(temp_dict.values())
    else:                                                                        # else determine missing
        temp = []
        i = 0
        for idx, image_link in temp_dict.items():
            if image_link is not None:
                output[i] = image_link
                i += 1
            else:
                temp.append(idx)

        print(f'Missing image IDs: {temp}')

    return output, idx_min, idx_max

def create_folders(mainOutputFolder, mainOutputSubFolders, idx_interval, do_prerun_output, pad_to):
    """
    function creates main project folder. -> descritive project name e.g [gallium_bubbles, water_bubbles]
    adds hierarchy of sub projects folders e.g [exp setup, parameter] 
    adds additional subfolder for subset of data e.g for images folder '0001-2999'
    so at this stage it looks like root-> exp setup -> parameter -> subset
    and output subfolders inside it for results and intermediate data
    in images output folder there are two folders. one is pre_run, for debugging initial state
    and one for final output
    generates file names for intermediate and aux data
    """
    interval_start, interval_stop = idx_interval
    if not os.path.exists(mainOutputFolder): os.mkdir(mainOutputFolder)  
    
    mainOutputSubFolders.append(f"{interval_start:0{pad_to}}-{interval_stop:0{pad_to}}")       # sub-project folder hierarhy e.g [exp setup, parameter, subset of data]

    for folderName in mainOutputSubFolders:     
        mainOutputFolder = os.path.join(mainOutputFolder, folderName)               
        if not os.path.exists(mainOutputFolder): os.mkdir(mainOutputFolder)


    images      = os.path.join(mainOutputFolder, 'images'    )
    stages      = os.path.join(mainOutputFolder, 'stages'    )
    archives    = os.path.join(mainOutputFolder, 'archives'  )
    graphs      = os.path.join(mainOutputFolder, 'graphs'    )

    [os.mkdir(folder) for folder in (images, stages, archives, graphs) if not os.path.exists(folder)]
    
    images_output = os.path.join(images, 'output')
    if not os.path.exists(images_output): os.mkdir(images_output)

    pre_run = None
    if do_prerun_output:
        pre_run = os.path.join(images, 'prerun')
        if not os.path.exists(pre_run): os.mkdir(pre_run)

    out_main = (images, stages, archives, graphs, images_output, pre_run)

    #mask at cropMaskPath (project -> setup -> parameter)
    cropMaskName = "-".join(mainOutputSubFolders[:2])+'-crop'
    cropMaskPath = os.path.join(os.path.join(*mainOutputFolder.split(os.sep)[:-1]), f"{cropMaskName}.png")
    cropMaskMissing = True if not os.path.exists(cropMaskPath) else False

    graphsPath          =   os.path.join(archives  ,  "graphs.pickle"    )
    segmentsPath        =   os.path.join(archives  ,  "segments.pickle"  )
    contoursHulls       =   os.path.join(archives  ,  "contorus.pickle"  )
    mergeSplitEvents    =   os.path.join(archives  ,  "ms-events.pickle" )

                        
    meanImagePath       =   os.path.join(archives  ,  "mean.npz"         )
    meanImagePathArr    =   os.path.join(archives  ,  "meanArr.npz"      )
                        
    archivePath         =   os.path.join(stages       ,  "croppedImageArr.npz"        )
    binarizedArrPath    =   os.path.join(stages       ,  "binarizedImageArr.npz"      )
    post_binary_data    =   os.path.join(stages       ,  "intermediate_data.pickle"   ) 

    out_aux = (cropMaskPath, cropMaskMissing, graphsPath, segmentsPath, contoursHulls, mergeSplitEvents, 
               meanImagePath, meanImagePathArr, archivePath, binarizedArrPath, post_binary_data)

    return out_main, out_aux
