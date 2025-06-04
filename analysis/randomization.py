"""analysis/randomization.py
Implement all the network link randomization methods"""

from numba import njit
from numba.typed.typedlist import List
import numpy as np

def compute_randomized(link_adjacency_matrix, mode, mc_iter=10):
    """Return a randomised version of the network respecting certain conditions regarding the mode chose:
    Modes:
    - `'i'`: "in" respect the in degree of each node
    - `'o'`: "out" respect the out degree of each node
    - `'i&&o'`: "in" and "out" respect the composite degree (both in and out degrees)
    - `'n-link'`: just reaffect links keeping its number constant
    - `"pn-link": reaffect links but keeping the number of total links and the proportion of reciprocal link constant`"""
    n = link_adjacency_matrix.shape[0]
    new_link_adjacency = np.zeros(link_adjacency_matrix.shape, dtype=int)

    if mode == "i&&o":
        return monte_carlo_randomisation(mc_iter, link_adjacency_matrix)
    elif mode == "n-link":
        return nlink_preserving_randomisation(link_adjacency_matrix)
    elif mode == "pn-link":
        return pnlink_preserving_randomisation(link_adjacency_matrix)
    elif mode == "i":
        return one_degree_preserving_randomisation(link_adjacency_matrix, "in")
    elif mode == "o":
        return one_degree_preserving_randomisation(link_adjacency_matrix, "out")
    else:
        raise KeyError("Unknown mode for randomisation. It must be in: 'o', 'i', 'i&&o', 'n-link', 'pn-link'")

def one_degree_preserving_randomisation(link_adjacency_matrix, degree):
    """Return a degree preserving randomisation of the network:
    Possible values for `degree`:
    - "in": Preserve in-degree
    - "out": Preserve out-degree"""
    n = link_adjacency_matrix.shape[0]
    new_link_adjacency = np.zeros(link_adjacency_matrix.shape, dtype=int)
    in_degree = np.sum(link_adjacency_matrix, axis=0)
    out_degree = np.sum(link_adjacency_matrix, axis=1)
    for i in range(n):
        if degree == "in":
            draw = np.random.choice(n-1, in_degree[i], replace=False) # n-1 because a link with oneself is not allowed
            for j in draw:
                if j < i:
                    new_link_adjacency[j, i] = 1
                else:
                    new_link_adjacency[j+1, i] = 1
        elif degree == "out":
            draw = np.random.choice(n-1, out_degree[i], replace=False)
            for j in draw:
                if j < i:
                    new_link_adjacency[i, j] = 1
                else:
                    new_link_adjacency[i, j+1] = 1
        else:
            raise KeyError("Unknown mode for one-degree preserving randomisation")
    return new_link_adjacency

def nlink_preserving_randomisation(link_adjacency_matrix):
    """Return a randomised version of the network with the same number of edges"""
    n = link_adjacency_matrix.shape[0]
    new_link_adjacency = np.zeros(link_adjacency_matrix.shape, dtype=int)
    total_link_number = int(np.sum(link_adjacency_matrix))
    try:
        draw_link = np.random.choice(n*(n-1), size=total_link_number, replace=False)
    except:
        raise Exception("Can't draw random links")
    draw_link.sort()
    for index in draw_link:
        i = index // (n-1)
        j_red = index % (n-1)
        if j_red >= i:
            j_red += 1
        new_link_adjacency[i, j_red] = 1
    return new_link_adjacency

def pnlink_preserving_randomisation(link_adjacency_matrix):
    """Return a randomised version of the network with the same number of edges
    and with the same proportion of reciprocal link"""
    n = link_adjacency_matrix.shape[0]
    new_link_adjacency = np.zeros(link_adjacency_matrix.shape, dtype=int)
    total_reciprocal_link = int(np.trace(link_adjacency_matrix @ link_adjacency_matrix)) // 2
    min_unidirectional_link = int(np.sum(link_adjacency_matrix)) - total_reciprocal_link
    link = []
    # Create only unidirectional link
    draw_unidirectionals = np.random.choice(n * (n-1) // 2, size=min_unidirectional_link, replace=False)
    for t in range(min_unidirectional_link):
        index = draw_unidirectionals[t]
        p = n-1
        i = 0
        j = index % p + 1
        while index // p > 0:
            i += 1
            index = index - p
            p -= 1
            j = i+1+index
        if np.random.random(1) > 0.5:
            drawn_link = (i, j)
        else:
            drawn_link = (j, i)
        new_link_adjacency[drawn_link[0], drawn_link[1]] = 1
        link.append(drawn_link)

    # Create reciprocal link using existing unidirectional links
    draw_reciprocal = np.random.choice(len(link), size=total_reciprocal_link, replace=False)
    for index in draw_reciprocal:
        drawn_link = link[index]
        new_link_adjacency[drawn_link[1], drawn_link[0]] = 1

    return new_link_adjacency
    
@njit
def monte_carlo_randomisation(niter, link_adjacency):
    """Randomise the network preserving both in and out degree"""

    new_link_adjacency = link_adjacency.copy()
    swappable_link = compute_swappable_links(new_link_adjacency)
    
    for it in range(niter):
        if len(swappable_link) == 0:
            print("ERROR: no more swappable link")
            print("Breaking loop at iteration: ", it)
            break

        draw = np.random.randint(len(swappable_link))
        links = swappable_link[draw]
        new_link_adjacency[links[2], links[1]] = new_link_adjacency[links[0], links[1]]
        new_link_adjacency[links[0], links[1]] = 0
        new_link_adjacency[links[0], links[3]] = new_link_adjacency[links[2], links[3]]
        new_link_adjacency[links[2], links[3]] = 0

        # Remove old edges
        i = 0
        while i < len(swappable_link): 
            test_links = swappable_link[i]
            if link_invalidate_test_link(test_links, links):
                swappable_link.pop(i) # Because pop is len dependent
            else:
                i += 1
        
        # Find new swappable
        swappable_links_from_link(links[2], links[1], new_link_adjacency, swappable_link)
        swappable_links_from_link(links[0], links[3], new_link_adjacency, swappable_link, other_link=(links[2], links[1]))
    
    return new_link_adjacency

@njit
def compute_swappable_links(link_adjacency):
    """Return all possible swappable link of the matrix conserving its structure"""
    n = link_adjacency.shape[0]
    swappable_link = List()
    new_link_adjacency = link_adjacency.copy()
    indexes = np.arange(n)
    for i in range(n):
        for j in range(i+1, n):
            if new_link_adjacency[i, j] > 0:
                c_selector = new_link_adjacency[i] < 1
                l_selector = new_link_adjacency[::,j] < 1
                for k in range(i+1):
                    l_selector[k] = False
                c_selector[i] = False
                l_selector[j] = False
                c_index = indexes[c_selector]
                l_index = indexes[l_selector]

                if len(l_index) == 0 or len(c_index) == 0:
                    continue

                for k in range(len(l_index)):
                    for l in range(len(c_index)):
                        if link_adjacency[l_index[k], c_index[l]] > 0:
                            swappable_link.append((i, j, l_index[k], c_index[l]))

    return swappable_link

@njit
def swappable_links_from_link(i, j, link_adjacency, swappable_link, other_link=(-1, -1)):
    """Add to `swappable_link` the swappable link containing the link i -> j in `link_adjacency`
    the parameter `other_link` is here to prevent doble link in swappable_link"""
    n = link_adjacency.shape[0]
    c_selector = link_adjacency[i] < 1
    l_selector = link_adjacency[::,j] < 1
    c_selector[i] = False
    l_selector[j] = False
    indexes = np.arange(n)
    c_index = indexes[c_selector]
    l_index = indexes[l_selector]
    
    if len(c_index) == 0 or len(l_index) == 0:
        return
    
    for k in range(len(l_index)):
        for l in range(len(c_index)):
            if l_index[k] == other_link[0] and c_index[l] == other_link[1]:
                continue
            elif link_adjacency[l_index[k], c_index[l]] > 0:
                swappable_link.append((i, j, l_index[k], c_index[l]))

@njit
def link_invalidate_test_link(test_links, links):
    """Return True if one of the change of link `links` invalidates `test_links`"""
    # Test for link that disappeared
    if links[0] == test_links[0] and links[1] == test_links[1]:
        return True
    elif links[0] == test_links[2] and links[1] == test_links[3]:
        return True
    elif links[2] == test_links[0] and links[3] == test_links[1]:
        return True
    elif links[2] == test_links[2] and links[3] == test_links[3]:
        return True
    # test for invalid XSWAP in-out square
    if links[2] == test_links[0] and links[1] == test_links[3]:
        return True
    elif links[2] == test_links[2] and links[1] == test_links[1]:
        return True
    if links[0] == test_links[0] and links[3] == test_links[3]:
        return True
    elif links[0] == test_links[2] and links[3] == test_links[1]:
        return True

    return False