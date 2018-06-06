# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:26:49 2015

@author: wangronin
"""

class func_wrap:
    
    def __init__(self, func):
        self.func = func

    def evaluate(self, x):
        return self.func(x)