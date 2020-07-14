import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams['figure.figsize'] = (3.0, 0.5) 
import matplotlib.patches as mpatches
from util.color_util import *
import pickle
from random import shuffle
import torch.optim as optim
import colorsys
from numpy import dot
from numpy.linalg import norm
from scipy import spatial
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from skimage import io, color
import random
from tabulate import tabulate

BASE_DIR = "../munroe"

def plot_color_single(color_in, fmt="rgb", save_path=None):
    """
    Plot a single color bar.

    Parameters
    ----------
    color_in : color tuple in rgb

    Returns
    -------
    None

    """
    ax = plt.gca()
    source = color_in.detach().numpy()
    if fmt != "rgb":
        source = colors.hsv_to_rgb(source)
    N = 100
    width, height = 1, 1
    for x in np.linspace(0, width, N):
        ax.add_patch(mpatches.Rectangle([x,0],width/N,height,color=np.clip(source,0,1)))
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# represent color hsv as in fourier space
def represent_color(color):
    h_s = color[0] * 1.0
    s_s = color[1] * 1.0
    v_s = color[2] * 1.0
    real_part = []
    imag_part = []
    for jj in range(0,3):
        for ii in range(0,3):
            for kk in range(0,3):
                comb = ii * h_s + jj * s_s + kk * v_s
                fft_comb = np.exp(-1.0 * 2j * np.pi * comb)
                real_part.append(fft_comb.real)
                imag_part.append(fft_comb.imag)
    real_part.extend(imag_part)
    ret = real_part

    return ret

# testing with some colors
def get_compara_direction(cdict, modifier, base_color, embeddings, net,
                          sample_per_color=1, fmt="rgb", fourier=False):
    '''
    compara: comparative color descriptions like `lighter`
    source_str: source color like `blue`
    '''
    
    # get colors!
    base_color = rand_choice(cdict[base_color], k=sample_per_color)
    base_color_in = []
    base_color_raw = []
    for b_c in base_color:
        b_c_in = b_c.tolist()
        if fourier:
            b_c_in = represent_color(b_c.tolist())
        base_color_in.append(b_c_in)           # feed in the complex version of hsvf
        base_color_raw.append(b_c.tolist())  # for loss use
    base_color_in = torch.tensor(base_color_in, dtype=torch.float)
    base_color_raw = torch.tensor(base_color_raw, dtype=torch.float)
    
    # get modifier setup
    if len(modifier) == 1:
        emb1, emb2 = \
            torch.zeros(sample_per_color, 300), \
            torch.tensor(embeddings[modifier[0]], dtype=torch.float)
        # repeat emb2 here
        emb2 = torch.stack([emb2]*sample_per_color, dim=0)
    else:
        emb1, emb2 = \
            torch.tensor(embeddings[modifier[0]], dtype=torch.float), \
            torch.tensor(embeddings[modifier[1]], dtype=torch.float)
        # repeat emb1,2 here
        emb1 = torch.stack([emb1]*sample_per_color, dim=0)
        emb2 = torch.stack([emb2]*sample_per_color, dim=0)
                
    wg = net(emb1, emb2, base_color_in)
    return base_color_raw.detach().numpy(), wg.detach().numpy()

def plot_color_change(cdict, compara, source_str, embeddings, net, strength=1,
                      save_path=None, sample_per_color=1, fmt="rgb",
                      fourier=False):
    '''
    compara: comparative color descriptions like `lighter`
    source_str: source color like `blue`
    strength: the ratio of exaggerating the effect to make it more perceivable
    '''
    source, direction = get_compara_direction(cdict, compara, source_str,
                                              net=net,
                                              sample_per_color=sample_per_color,
                                              embeddings=embeddings,
                                              fmt="rgb", fourier=False)
    for i in range(sample_per_color):
        ax = plt.gca()
        N = 100
        width, height = 1, 1
        for x in np.linspace(0, width, N):
            hsv = source[i]+direction[i]*x*strength
            if fmt != rgb:
                rgb = colors.hsv_to_rgb(hsv)
            ax.add_patch(mpatches.Rectangle([x,0],width/N,height,color=np.clip(rgb,0,1)))
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.show()

def plot_color_change_raw(source, direction, strength=1, save_path=None,
                          fmt="rgb"):
    '''
    compara: comparative color descriptions like `lighter`
    source_str: source color like `blue`
    strength: the ratio of exaggerating the effect to make it more perceivable
    '''
    ax = plt.gca()
    N = 100
    width, height = 1, 1
    for x in np.linspace(0, width, N):
        c = source+direction*x*strength
        if fmt != "rgb":
            c = colors.hsv_to_rgb(c)
        ax.add_patch(mpatches.Rectangle([x,0],width/N,height,color=np.clip(c,0,1)))
    print(c)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

'''
Helper function
TODO: Move all of these into a util function
'''
def load_splits(test_dev=True):
    train, dev, test = [], [], []
    basedir = os.path.join(BASE_DIR, "xkcd_colordata")
    for cfile in os.listdir(basedir):
        if cfile.endswith(".train"):
            train.append(cfile)
        elif cfile.endswith(".dev"):
            if test_dev:
                dev.append(cfile)
                test.append(cfile)
            else:
                dev.append(cfile)
        elif cfile.endswith(".test"):
            test.append(cfile)
    return train, dev, test

def load_rgb(split):
    cdict = {}
    basedir = os.path.join(BASE_DIR, "xkcd_colordata")
    for cfile in split:
        cname = cfile[:cfile.find(".")]
        with open(os.path.join(basedir, cfile),"rb") as f:
            all_smaples = np.array(pickle.load(f)) / 255   # normalized to [0,1]
            cdict[cname] = torch.FloatTensor(all_smaples)
    return cdict

def load_triple(cdict, label2words, extend=True):
    """
    Loading triples of color modifiers

    Parameters
    ----------
    cdict : dict
        Color dictionary maps a string to list of rgb tuples.
    label2words : dict
        Dictionary mapping color labels to color names.

    Returns
    -------
    dict:
        Triples can be formed by colors in the cdict dictionary.

    """
    bypass_quantifier = ["almost","cobalt"]
    file_comp = os.path.join(BASE_DIR, "comparatives.txt")
    quan_comp = os.path.join(BASE_DIR, "quantifiers.txt")
    to_compara = dict(line.strip().split(":") for line in open(file_comp, encoding="utf-8"))
    to_more_quanti = dict(line.strip().split(":") for line in open(quan_comp, encoding="utf-8"))
    
    triples = []
    for label in cdict:
        words = label2words[label].split()
        if len(words) > 1:
            quantifier, base = words[0], "".join(words[1:])
            if quantifier == "very":
                base = "".join(words[2:])
                quantifier = words[1]
                if base in cdict:
                    if words[1] in to_compara:
                        triples.append((base, ("more", to_compara[quantifier]), tuple(label2words[base].split()), label))
            else:
                if base in cdict:
                    if quantifier in to_compara:        # uni-gram('lighter',)
                        triples.append((base, (to_compara[quantifier],), tuple(label2words[base].split()), label))
                    elif quantifier in to_more_quanti:  # bigram('more','bluish')
                        triples.append((base, ("more", to_more_quanti[quantifier]), tuple(label2words[base].split()), label))
                    else:
                        if extend:
                            # this adds more power, but not increase AUC
                            if quantifier not in bypass_quantifier:
                                triples.append((base, ("more", quantifier), tuple(label2words[base].split()), label))
    return triples

def glove2dict(src_filename):
    """
    GloVe reader.

    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.

    Returns
    -------
    dict
        Mapping words to their GloVe vectors as `np.array`.

    """
    # This distribution has some words with spaces, so we have to
    # assume its dimensionality and parse out the lines specially:
    if '840B.300d' in src_filename:
        line_parser = lambda line: line.rsplit(" ", 300)
    else:
        line_parser = lambda line: line.strip().split()
    data = {}
    with open(src_filename, encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line_parser(line)
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def rand_choice(tensor_in, k=10, dim=0):
    if k == 1:
        # this case, we will take the mean
        return tensor_in.mean(dim=dim).unsqueeze(dim=dim)
    # to make our random sample more consistent with the mean, 
    # we are taking a bootstrapping approach to get the sample.
    # we will randomly select k mean samples of a sampled set.
    dim_c = tensor_in.size(dim)
    bs_c = 30
    samples = []
    for i in range(k):
        perm = torch.randperm(dim_c)
        idx = perm[:bs_c]
        sample = tensor_in[idx].mean(dim=dim)
        samples.append(sample)
    return torch.stack(samples, dim=0)

def generate_test_set(triple_train, triple_test):
    # generate different sets from triple_train and triple_test
    seen_pair = set([])
    for pair in triple_test:
        if pair in triple_train:
            seen_pair.add(pair)

    # sets to compare with
    base_color = set([])
    modifier = set([])
    target_color = set([])
    for pair in triple_train:
        base_color.add(pair[0])
        modifier.add(pair[1])
        target_color.add(pair[-1])
    
    unseen_pair = set([])
    for pair in triple_test:
        if pair not in triple_train:
            if pair[0] in base_color and pair[1] in modifier:
                unseen_pair.add(pair)
                
    unseen_base = set([])
    for pair in triple_test:
        if pair not in triple_train:
            if pair[1] in modifier and pair[0] not in base_color:
                unseen_base.add(pair)

    unseen_mod = set([])
    for pair in triple_test:
        if pair not in triple_train:
            if pair[0] in base_color and pair[1] not in modifier:
                unseen_mod.add(pair)
                
    unseen_fully = set([])
    for pair in triple_test:
        if pair not in triple_train:
            if pair[0] not in base_color and pair[1] not in modifier:
                unseen_fully.add(pair)
    
    return {"seen_pair" : seen_pair,
            "unseen_pair" : unseen_pair,
            "unseen_base" : unseen_base,
            "unseen_mod" : unseen_mod,
            "unseen_fully" : unseen_fully,
            "overall" : triple_test}

def generate_test_set_inverse(triple_train, triple_test):
    # generate different sets from triple_train and triple_test
    seen_pair = set([])
    for pair in triple_test:
        if pair in triple_train:
            seen_pair.add(pair)

    # sets to compare with
    base_color = set([])
    modifier = set([])
    target_color = set([])
    for pair in triple_train:
        base_color.add(pair[-1])
        modifier.add(pair[1])
        target_color.add(pair[0])
    
    unseen_pair = set([])
    for pair in triple_test:
        if pair not in triple_train:
            if pair[-1] in base_color and pair[1] in modifier:
                unseen_pair.add(pair)
                
    unseen_base = set([])
    for pair in triple_test:
        if pair not in triple_train:
            if pair[1] in modifier and pair[-1] not in base_color:
                unseen_base.add(pair)

    unseen_mod = set([])
    for pair in triple_test:
        if pair not in triple_train:
            if pair[-1] in base_color and pair[1] not in modifier:
                unseen_mod.add(pair)
                
    unseen_fully = set([])
    for pair in triple_test:
        if pair not in triple_train:
            if pair[-1] not in base_color and pair[1] not in modifier:
                unseen_fully.add(pair)
    
    return {"seen_pair" : seen_pair,
            "unseen_pair" : unseen_pair,
            "unseen_base" : unseen_base,
            "unseen_mod" : unseen_mod,
            "unseen_fully" : unseen_fully,
            "overall" : triple_test}

'''
Standard evaluation function that evaluate
any models with the test set result.

We will use rgb format when evaluating, as
this is the way that previous works are using.

Return each color the mean metrics.
'''
def evaluate_color(net_predict, fmt="rgb", eval_target="pred", reduced=False):
    evaluation_metrics = dict()
    for k in net_predict:
        evaluation_metrics[k] = dict()
        # we have 2 metrices to report
        evaluation_metrics[k]["cosine"] = []
        evaluation_metrics[k]["delta_E"] = []
        for triple in net_predict[k].keys():
            true = net_predict[k][triple]["true"]
            pred = net_predict[k][triple][eval_target]
            base = net_predict[k][triple]["base"]
            if reduced:
                pred = pred.mean(dim=0).unsqueeze(dim=0)
            sample_size = pred.shape[0]
            color_sim = 0.0
            color_delta_e = 0.0
            for i in range(sample_size):
                if fmt == "rgb":
                    pred_rgb = pred[i]
                    true_rgb = true[0]
                    base_rgb = base[0]
                else:
                    pred_rgb = torch.tensor(colors.hsv_to_rgb(pred[i]))  # rgb space for target color
                    true_rgb = torch.tensor(colors.hsv_to_rgb(true[0]))  # keep consistent with previous paper
                    base_rgb = torch.tensor(colors.hsv_to_rgb(base[0]))  # rgb space for target color
                # cosine metrics
                cos_sim = 1 - spatial.distance.cosine(pred_rgb - base_rgb, true_rgb - base_rgb)
                color_sim += cos_sim
                # delta_E
                c1 = sRGBColor(rgb_r=pred_rgb[0], rgb_g=pred_rgb[1], rgb_b=pred_rgb[2])
                c2 = sRGBColor(rgb_r=true_rgb[0], rgb_g=true_rgb[1], rgb_b=true_rgb[2])
                # Convert from RGB to Lab Color Space
                color1_lab = convert_color(c1, LabColor)
                # Convert from RGB to Lab Color Space
                color2_lab = convert_color(c2, LabColor)
                delta_e = delta_e_cie2000(color1_lab, color2_lab)
                color_delta_e += delta_e

            color_sim = color_sim*1.0 / sample_size  # color avg cosine
            color_delta_e = color_delta_e*1.0 / sample_size  # color avg cosine
            
            evaluation_metrics[k]["cosine"].append(color_sim)
            evaluation_metrics[k]["delta_E"].append(color_delta_e)
            
    # display evaluation metrices accordingly
    display_list = []
    for condition in evaluation_metrics.keys():
        cosine = evaluation_metrics[condition]["cosine"]
        delta_E = evaluation_metrics[condition]["delta_E"]
        cosine_str = "%s (%s)" % ('{:.3f}'.format(np.mean(cosine)), '{:.3f}'.format(np.std(cosine), ddof=1))
        delta_E_str = "%s (%s)" % ('{:.3f}'.format(np.mean(delta_E)), '{:.3f}'.format(np.std(delta_E), ddof=1))
        row = [condition, cosine_str, delta_E_str]
        display_list.append(row)
        
    print(tabulate(display_list, headers=['condition', 'cosine (std)', 'delta_E (std)']))
    
    return evaluation_metrics

'''
Batch Generation Function
'''
def generate_batch(cdict, triple, embeddings, batch_size_in_tuple=50, sample_per_color=1, eval=False,
                   fourier=True, listener=False):
    input_size = len(triple)
    index = [i for i in range(0, input_size)]
    if not eval:
        shuffle(index)
    base_index = 0
    target_index = -1
    if listener:
        base_index = -1
        target_index = 0
    shuffle_chunks = [i for i in chunks(index, batch_size_in_tuple)]
    for chunk in shuffle_chunks:
        # batch data in
        batch_emb1 = []
        batch_emb2 = []
        batch_base_color = []
        batch_base_color_raw = []
        batch_target_color = []
        eval_only_target_color = []
        eval_only_base_color = []

        # other metadata will also be saved
        batch_triple = []

        for index in chunk:
            base_color = \
                rand_choice(cdict[triple[index][base_index]],
                            k=sample_per_color)
            batch_triple.append(triple[index])
            base_color_in = []
            base_color_raw = []
            for b_c in base_color:
                if fourier:
                    b_c_process = represent_color(b_c.tolist())
                else:
                    b_c_process = b_c.tolist()       # just raw hsv data
                base_color_in.append(b_c_process)    # using complex version
                base_color_raw.append(b_c.tolist())  # for loss use
            base_color_in = torch.tensor(base_color_in, dtype=torch.float)
            base_color_raw = torch.tensor(base_color_raw, dtype=torch.float)
            target_color = \
                rand_choice(cdict[triple[index][target_index]],
                            k=sample_per_color)
            eval_target_color = \
                rand_choice(cdict[triple[index][target_index]], k=1)
            eval_base_color = rand_choice(cdict[triple[index][base_index]], k=1)
            modifier = triple[index][1]
            if len(modifier) == 1:
                emb1, emb2 = \
                    torch.zeros(sample_per_color, 300), \
                    torch.tensor(embeddings[modifier[0]], dtype=torch.float)
                # repeat emb2 here
                emb2 = torch.stack([emb2]*sample_per_color, dim=0)
            else:
                emb1, emb2 = \
                    torch.tensor(embeddings[modifier[0]], dtype=torch.float), \
                    torch.tensor(embeddings[modifier[1]], dtype=torch.float)
                # repeat emb1,2 here
                emb1 = torch.stack([emb1]*sample_per_color, dim=0)
                emb2 = torch.stack([emb2]*sample_per_color, dim=0)
                
            batch_emb1.append(emb1)
            batch_emb2.append(emb2)
            batch_base_color.append(base_color_in)
            batch_base_color_raw.append(base_color_raw)
            batch_target_color.append(target_color)
            eval_only_target_color.append(eval_target_color)
            eval_only_base_color.append(eval_base_color)
        
        # consolidate
        batch_emb1 = torch.cat(batch_emb1, dim=0)
        batch_emb2 = torch.cat(batch_emb2, dim=0)
        batch_base_color = torch.cat(batch_base_color, dim=0)
        batch_base_color_raw = torch.cat(batch_base_color_raw, dim=0)
        batch_target_color = torch.cat(batch_target_color, dim=0)
        eval_only_target_color = torch.cat(eval_only_target_color, dim=0)
        eval_only_base_color = torch.cat(eval_only_base_color, dim=0)
        
        if eval:
            yield batch_emb1, batch_emb2, batch_base_color, \
                batch_base_color_raw, batch_target_color, \
                eval_only_target_color, eval_only_base_color, batch_triple
        else:
            # colors are all in hsv space not rgb space!
            yield batch_emb1, batch_emb2, batch_base_color, \
                batch_base_color_raw, batch_target_color

'''
Prediction function that will generate predictions using the 
pretrained network passed in on the triples provided. The 
triple is a dictionary if there are multiple sets.

Return a dictionary containing on the materials evaluation
needed.
'''
def predict_color(net, triple, cdict, embeddings, sample_per_color=1,
                  fourier=False, listener=False):
    predict = dict()
    # with eval mode
    with torch.no_grad():
        net.eval()
        for k in triple.keys():
            predict[k] = dict()
            print("predict %s set with %d samples." % (k, len(triple[k])))
            triple_set = list(triple[k])
            for batch_emb1, batch_emb2, \
                batch_base_color, batch_base_color_raw, \
                batch_target_color, eval_only_target_color, eval_only_base_color, \
                batch_triple in \
                generate_batch(cdict, triple_set, embeddings, batch_size_in_tuple=1, 
                               sample_per_color=sample_per_color, eval=True,
                               fourier=fourier, listener=listener):
                pred = net(batch_emb1, batch_emb2, batch_base_color)
                assert(len(batch_triple) == 1)
                predict[k][batch_triple[0]] = dict()
                predict[k][batch_triple[0]]["true"] = eval_only_target_color
                predict[k][batch_triple[0]]["pred"] = pred
                predict[k][batch_triple[0]]["s_true"] = batch_target_color
                predict[k][batch_triple[0]]["s_base"] = batch_base_color_raw
                predict[k][batch_triple[0]]["base"] = eval_only_base_color
                predict[k][batch_triple[0]]["embeddings_1"] = batch_emb1
                predict[k][batch_triple[0]]["embeddings_2"] = batch_emb2
    return predict

def predict_color_hsv(net, triple, cdict, embeddings, sample_per_color=1, fourier=False):
    predict = dict()
    # with eval mode
    with torch.no_grad():
        net.eval()
        for k in triple.keys():
            predict[k] = dict()
            print("predict %s set with %d samples." % (k, len(triple[k])))
            triple_set = list(triple[k])
            for batch_emb1, batch_emb2, \
                batch_base_color, batch_base_color_raw, \
                batch_target_color, eval_only_target_color, eval_only_base_color, \
                batch_triple in \
                generate_batch(cdict, triple_set, embeddings, batch_size_in_tuple=1, 
                               sample_per_color=sample_per_color, eval=True, fourier=fourier):
                h_pred, sv_pred = net(batch_emb1, batch_emb2, batch_base_color)
                pred_hsv = torch.cat([h_pred, sv_pred], dim=-1)
                assert(len(batch_triple) == 1)
                predict[k][batch_triple[0]] = dict()
                predict[k][batch_triple[0]]["pred"] = pred_hsv.mean(dim=0).unsqueeze(dim=0)
                predict[k][batch_triple[0]]["true"] = eval_only_target_color
                predict[k][batch_triple[0]]["base"] = eval_only_base_color
                predict[k][batch_triple[0]]["s_pred"] = pred_hsv
                predict[k][batch_triple[0]]["s_true"] = batch_target_color
                predict[k][batch_triple[0]]["s_base"] = batch_base_color_raw
                
    return predict

def softmax_prob(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def pragmatic_predict(net_predict, level="s1"):
    for set_name in net_predict.keys():
        for triple in net_predict[set_name].keys():
            sample_cos = []
            sample_delta_e = []
            for sample in net_predict[set_name][triple]["pred"]:
                pred_sample = sample.detach().numpy()
                base = net_predict[set_name][triple]["base"].detach().numpy()[0]
                # prob based on cosine
                cos_dis = spatial.distance.cosine(pred_sample - base, base)
                sample_cos.append(cos_dis)
                # prob based on delta_e
                c1 = sRGBColor(rgb_r=pred_sample[0], rgb_g=pred_sample[1], rgb_b=pred_sample[2])
                c2 = sRGBColor(rgb_r=base[0], rgb_g=base[1], rgb_b=base[2])
                # Convert from RGB to Lab Color Space
                color1_lab = convert_color(c1, LabColor)
                # Convert from RGB to Lab Color Space
                color2_lab = convert_color(c2, LabColor)
                delta_e = delta_e_cie2000(color1_lab, color2_lab)
                sample_delta_e.append(delta_e)

            cos_prob = softmax_prob(sample_cos)
            delta_e_prob = softmax_prob(sample_delta_e)
            combine_prob = np.multiply(cos_prob, delta_e_prob)
            combine_prob = softmax_prob(combine_prob)

            cos_prob_n = np.argmax(cos_prob)
            delta_e_prob_n = np.argmax(delta_e_prob)
            combine_prob_n = np.argmax(combine_prob)

            all_sample = net_predict[set_name][triple]["pred"]
            net_predict[set_name][triple]["pred_cos_s1"] = \
                all_sample[cos_prob_n].unsqueeze(dim=0)
            net_predict[set_name][triple]["pred_delta_e_s1"] = \
                all_sample[delta_e_prob_n].unsqueeze(dim=0)
            net_predict[set_name][triple]["pred_combine_s1"] = \
                all_sample[combine_prob_n].unsqueeze(dim=0)

            pred_weight = None
            index = 0
            for prob in delta_e_prob:
                if pred_weight is None:
                    pred_weight = prob * all_sample[index]
                else:
                    pred_weight += prob * all_sample[index]
                index += 1

            net_predict[set_name][triple]["pred_weight_s1"] = \
                pred_weight.unsqueeze(dim=0)
    return net_predict