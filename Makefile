# Makefile for GPT-2 C++ Inference
# 
# Build: make
# Debug: make DEBUG=1
# Clean: make clean

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

# Source files
SRCS = src/main.cpp src/ops.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = gpt2

# Build mode
ifdef DEBUG
	CXXFLAGS += -g -O0 -fsanitize=address
	LDFLAGS += -fsanitize=address
else ifdef FAST
	CXXFLAGS += -O3 -march=native -ffast-math
else
	CXXFLAGS += -O2
endif

# Targets
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)