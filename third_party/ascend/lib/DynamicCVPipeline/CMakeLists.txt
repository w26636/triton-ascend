file(GLOB PLAN_COMPUTE_BLOCK_SRC_FILES CONFIGURE_DEPENDS
    "PlanComputeBlock/*.cpp"
)

file(GLOB ADD_CONTROLFLOW_CONDITION_SRC_FILES CONFIGURE_DEPENDS
 	     "AddControlFlowCondition/*.cpp"
)

file(GLOB SEPARATE_MEMORY_FROM_COMPUTE_SRC_FILES CONFIGURE_DEPENDS
  "SeparateMemoryFromCompute/*.cpp"
)

add_triton_library(DynamicCVPipeline
  AddDynamicCVPipeline.cpp
  PlanComputeBlock.cpp
  SplitDataflow.cpp
  SeparateMemoryFromComputePass.cpp
  AllocMultiCache.cpp
  AddControlFlowCondition.cpp
  ${PLAN_COMPUTE_BLOCK_SRC_FILES}
  ${ADD_CONTROLFLOW_CONDITION_SRC_FILES}
  ${SEPARATE_MEMORY_FROM_COMPUTE_SRC_FILES}

  DEPENDS
  AddDynamicCVPipelinePassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
)