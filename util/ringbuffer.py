#!/usr/bin/env python
'''
Copyright (c) 2017, Jesper Derehag <jderehag@hotmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import numpy as np

class RingBuffer(object):
    '''
    TODO: Write docstring
    '''
    def __init__(self, size_max, dtype=object):
        '''
        TODO: Write docstring
        '''
        self.size = 0
        self.size_max = size_max
        self._data = np.empty(size_max, dtype=dtype)

    def append(self, value):
        '''
        TODO: Write docstring
        '''
        self._data = np.roll(self._data, 1)
        self._data[0] = value

        if self.size < self.size_max:
            self.size += 1

    def sample(self, size):
        '''
        TODO: Write docstring
        '''
        data = self._data if self.size == self.size_max else self._data[0:self.size]
        return np.random.choice(data, size=size, replace=False)

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = repr(self._data)
        s += '\t' + repr(self.size)
        s += '\t' + repr(self._data[::-1])
        s += '\t' + repr(self._data[0:self.size][::-1])
        return s
