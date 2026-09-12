NVCC := nvcc

NVCCFLAGS := -std=c++17 -O3 -Iinclude

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

TARGET := $(BIN_DIR)/gemm

CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)

CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))

OBJS := $(CPP_OBJS) $(CU_OBJS)
DEPS := $(OBJS:.o=.d)
DEPFLAGS = -MMD -MF $(@:.o=.d) -MT $@

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(OBJS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp Makefile | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(DEPFLAGS) -x c++ -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu Makefile | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(DEPS)
