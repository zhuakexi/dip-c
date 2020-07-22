import sys
import getopt
import copy
import rmsd
import numpy as np

#input_data -> common_data

THRESHOLD=2
def do_pick(pairs:list,dv:list):
    #pick out structure dv bigger than THRESHOLD, pairs: all combinations in order, dv: median_devariations
    ##get all structure name(represent by int index) 
    problematic = []
    [problematic.extend(i) for i in pairs]
    problematic = set(problematic)
    ##do the pick
    for i,v in enumerate(dv):
        if v < THRESHOLD:
            problematic -= pairs[i]
    #print(problematic)
    ##get index of good pairs
    good_pairs = [i for i,j in enumerate(pairs) if len(j.intersection(problematic)) == 0]
    return problematic, good_pairs
def align(argv):
    # default parameters
    output_prefix = None

    # read arguments
    try:
        opts, args = getopt.getopt(argv[1:], "o:")
    except getopt.GetoptError as err:
        sys.stderr.write("[E::" + __name__ + "] unknown command\n")
        return 1
    if len(args) == 0:
        sys.stderr.write("Usage: dip-c align [options] <in1.3dg> <in2.3dg> ...\n")
        sys.stderr.write("Options:\n")
        sys.stderr.write("  -o STR        output prefix [no output]\n")
        sys.stderr.write("Output:\n")
        sys.stderr.write("  tab-delimited: homolog, locus, RMSD\n")
        sys.stderr.write("  additionally with \"-o\": 3DG files aligned to each other\n")

        return 1
    for o, a in opts:
        if o == "-o":
            output_prefix = a            
            
    # ------------------------load 3dg files--------------------------
    input_data = []
    num_structures = len(args)
    if num_structures < 2:
        sys.stderr.write("[E::" + __name__ + "] at least 2 structures are required\n")
        return 1
    counter = 0
    input_filenames = args
    for input_filename in input_filenames:
        sys.stderr.write("[M::" + __name__ + "] reading 3dg file " + str(counter) + ": " + input_filename + "\n")
        input_data.append({})
        for input_file_line in open(input_filename, "rb"):
            input_file_line_data = input_file_line.strip().split()
            #store in the newly added empty dictionary
            input_data[-1][(input_file_line_data[0], int(input_file_line_data[1]))] = [float(input_file_line_data[2]),float(input_file_line_data[3]),float(input_file_line_data[4])]
        counter += 1
    #--------------------------------find common particles--------------------------------
    # find common particles
    common_loci = set(input_data[0])
    ##get all keys
    for input_structure in input_data[1:]:
        common_loci = common_loci.intersection(set(input_structure))
    num_loci = len(common_loci)
    common_loci = list(common_loci)
    common_data = []
    for input_structure in input_data:
        common_data.append([])
        for common_locus in common_loci:
            common_data[-1].append(input_structure[common_locus])
    #select data subset for each structure according to common_loci
    sys.stderr.write("[M::" + __name__ + "] found " + str(num_loci) + " common particles\n")
    
    #common_data = np.array(common_data)
    # --------------------subtract centroid---------------------
    common_data = np.array(common_data)
    centroid_data = []
    #normalize to centroid for each structure
    for i in range(num_structures):
        common_data[i] = np.array(common_data[i])
        centroid_pos = rmsd.centroid(common_data[i])
        common_data[i] -= centroid_pos
        centroid_data.append(centroid_pos)
    sys.stderr.write("[M::" + __name__ + "] found centroids for " + str(num_structures) + " structures\n")
    # ------------calculate pairwise deviation and rotate------------
    #for deviations: locus:sturcture
    deviations = np.empty((num_loci, 0), float)
    #for pick up
    median_deviations = []
    dv_pairs = []
    for i in range(num_structures):
        for j in range(num_structures):
            if j == i:
                continue
            # mirror image if needed
            mirror_factor = 1.0
            if rmsd.kabsch_rmsd(common_data[i], common_data[j]) > rmsd.kabsch_rmsd(common_data[i], -1.0 * common_data[j]):
                mirror_factor = -1.0
            # calculate deviation
            rotation_matrix = rmsd.kabsch(mirror_factor * common_data[j], common_data[i])
            if j > i:
                #print("matrix",np.dot(mirror_factor * common_data[j], rotation_matrix) - common_data[i])
                deviation = np.linalg.norm(np.dot(mirror_factor * common_data[j], rotation_matrix) - common_data[i], axis = 1).T
                #print("deviation",deviation)
                deviations = np.c_[deviations, deviation]
                #print("deviations",deviations)
                dv_pairs.append( set([i,j]) )
                median_deviations.append(np.median(deviation))
                sys.stderr.write("[M::" + __name__ + "] median deviation between file " + str(i) + " and file " + str(j) + ": " + str(np.median(deviation)) + "\n")
            
            # rotate
            if output_prefix is not None:
                # rotate j to align with i
                sys.stderr.write("[M::" + __name__ + "] aligning file " + str(j) + " to file " + str(i) + "\n")
                aligned_filename = output_prefix + str(j) + "_to_" + str(i) + ".3dg"
                aligned_file = open(aligned_filename, "wb")
                for input_locus in input_data[j]:
                    aligned_pos = np.dot((np.array(input_data[j][input_locus]) - centroid_data[j]) * mirror_factor, rotation_matrix) + centroid_data[i]
                    aligned_file.write("\t".join([input_locus[0], str(input_locus[1]), str(aligned_pos[0]), str(aligned_pos[1]), str(aligned_pos[2])]) + "\n")
                aligned_file.close()
    # ------------calculate rmsds------------

    # ------------exclude structure deviation bigger than threthold------------
    problematic, good_pairs = do_pick(dv_pairs, median_deviations)
    if len(problematic) > 0:
        exclude_files = [input_filenames[i] for i in problematic]
        good_deviations = [deviations[:,i] for i in good_pairs]
        good_deviations = np.array(good_deviations).T
        #print("deviations", deviations.shape)
        #print("good_deviations", good_deviations.shape)
        deviations = good_deviations   
    #print(dv_pairs)
    #print(median_deviations)
    # ------------summarize rmsd and print------------
    rmsds = np.sqrt((deviations ** 2).mean(axis = 1))
    #print("rmsds", rmsds.shape)
    totalrmsd = np.sqrt((rmsds ** 2).mean(axis = 0))
    
    sys.stderr.write( "[M::" + __name__ + "] exclude: " + str(exclude_files) +"\n")
    #RMS RMSD rmsds -> square -> median -> aqrt
    sys.stderr.write("[M::" + __name__ + "] RMS RMSD: " + str(totalrmsd) + "\n")
    # median RMSD rmsds -> median 
    sys.stderr.write("[M::" + __name__ + "] median RMSD: " + str(np.median(rmsds,axis = 0)) + "\n")
    sys.stderr.write("[M::" + __name__ + "] writing output\n")
    
    for i in range(num_loci):
        sys.stdout.write("\t".join(map(str, [common_loci[i][0], common_loci[i][1], rmsds[i]])) + "\n")
    return 0
    