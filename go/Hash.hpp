// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "go.hpp"

typedef uint64_t Hash;

Hash hashOf(uint64_t black, uint64_t white, int codedKo, bool pass, bool swapped);
