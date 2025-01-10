import numpy as np
from collections import deque
from .decas import *
import heapq

class Graph:
    def __init__(self, segs):
        #segs: List of segments
        self.__ngr = {} # non-direct graph
        self.__gr = {}
        # visited = []
        # print(len(segs))
        for s in segs:
            seg_data= s.get_segment()
            self.__gr[seg_data[0]] = [s]
            self.__ngr[seg_data[0]] = [s]
            for s_ in segs:
                seg_data_= s_.get_segment()
                if s_!=s and seg_data_[0] == seg_data[0]:
                    self.__gr[seg_data[0]].append(s_)
                    self.__ngr[seg_data[0]].append(s_)
        for s in segs:
            seg_data= s.get_segment()
            if seg_data[1] not in self.__ngr.keys():
                self.__ngr[seg_data[1]] = [s]
            for s_ in segs:
                seg_data_= s_.get_segment()
                if s_!=s and seg_data_[1] == seg_data[1]:
                    self.__ngr[seg_data[1]].append(s_)
        print("finish build a graph from segment information")
    
    def get_vertexes(self, direction = False):
        if direction:
            return self.__gr.keys()
        return self.__ngr.keys()
    
    def dijkstra(self, pnt1 :Point, pnt2:Point):
        vertexes = self.__ngr.keys()
        dijkstra_ = {}
        for ver in  vertexes:
            dijkstra_[ver] = (float('inf'), "n")
        dijkstra_[pnt1] = (0, pnt1)
        queue = [pnt1]

        visited = []
        while(queue):
            cur_chkpnt = queue.pop(0)
            if cur_chkpnt in visited:
                continue
            edges = self.__ngr[cur_chkpnt]
            for ed in edges:
                pnt = ed.get_segment()
                if pnt[0] == cur_chkpnt:
                    queue.append(pnt[1])
                    pnt_vl = dijkstra_[cur_chkpnt][0] + ed.get_long()
                    if(dijkstra_[pnt[1]][0] > pnt_vl):
                        dijkstra_[pnt[1]] = (pnt_vl,cur_chkpnt)
                elif pnt[1] == cur_chkpnt:
                    queue.append(pnt[0])
                    pnt_vl = dijkstra_[cur_chkpnt][0] + ed.get_long()
                    if(dijkstra_[pnt[0]][0] > pnt_vl):
                        dijkstra_[pnt[0]] = (pnt_vl,cur_chkpnt)
            visited.append(cur_chkpnt)
        paths = [pnt2]
        checkpnt = pnt2
        cnt = 0
        self.__minlong = dijkstra_[pnt2]
        while(checkpnt!= pnt1):
            checkpnt = dijkstra_[checkpnt][1]
            if checkpnt not in paths:
                paths.insert(0, checkpnt)
            cnt += 1
            if cnt > len(dijkstra_):
                raise ValueError("the input point in the map is in correct")
            
        return paths
    def get_min_long(self):
        return self.__minlong
    def get_graph(self, type = 'N'):
        if type =='N':
            return self.__ngr
        else:
            return self.__gr
        
    def get_possible_roots(self, ver_dp, ver_des, type='N'):
        #type N: undirect grap
        #type D: direct grap
        graph = self.__ngr if type == 'N' else self.__gr
        queue = deque()
        possible_roots = []

        # Start from all segments that begin with ver_dp
        if ver_dp in graph:
            for segment in graph[ver_dp]:
                queue.append((ver_dp, segment, [segment]))

        while queue:
            current_ver_dp, current_segment, path = queue.popleft()
            
            current_end = current_segment.get_segment()
            if current_ver_dp==current_end[0]:
                current_end=current_end[1]
            elif current_ver_dp==current_end[1]:
                current_end=current_end[0]
            
            # If we reach the destination, add the first segment of this path to possible_roots
            if current_end == ver_des and path not in possible_roots:
                possible_roots.append(path)
            
            for next_segment in graph.get(current_end, []):
                if next_segment not in path:  # Avoid cycles
                    queue.append((current_end ,next_segment, path + [next_segment]))
        return possible_roots
    
    def get_direct_possible_roots(self, ver_dp, ver_des, dir = "LR"):
        pass
    
    def get_undirect_possible_roots(self, ver_dp, ver_des):
        pass
    
        
        
        
        
    
                        
                        