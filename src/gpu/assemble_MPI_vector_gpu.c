/*
 !=====================================================================
 !
 !          S p e c f e m 3 D  G l o b e  V e r s i o n  6 . 0
 !          --------------------------------------------------
 !
 !     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
 !                        Princeton University, USA
 !                and CNRS / University of Marseille, France
 !                 (there are currently many more authors!)
 ! (c) Princeton University and CNRS / University of Marseille, April 2014
 !
 ! This program is free software; you can redistribute it and/or modify
 ! it under the terms of the GNU General Public License as published by
 ! the Free Software Foundation; either version 2 of the License, or
 ! (at your option) any later version.
 !
 ! This program is distributed in the hope that it will be useful,
 ! but WITHOUT ANY WARRANTY; without even the implied warranty of
 ! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 ! GNU General Public License for more details.
 !
 ! You should have received a copy of the GNU General Public License along
 ! with this program; if not, write to the Free Software Foundation, Inc.,
 ! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 !
 !=====================================================================
*/

#include <string.h>

#include "mesh_constants_gpu.h"
/* ----------------------------------------------------------------------------------------------- */
// MPI transfer
/* ----------------------------------------------------------------------------------------------- */


// prepares and transfers the inter-element edge-nodes to the host to be MPI'd
// (elements on boundary)

extern EXTERN_LANG
void FC_FUNC_(transfer_boun_from_device,
              TRANSFER_BOUN_FROM_DEVICE)(long *Mesh_pointer_f,
                                                 realw *send_accel_buffer,
                                                 int *IREGION,
                                                 int *FORWARD_OR_ADJOINT) {
  TRACE("transfer_boun_from_device");

  int blocksize, size_padded;
  int num_blocks_x, num_blocks_y;

#ifdef USE_OPENCL
  size_t global_work_size[2];
  size_t local_work_size[2];
  cl_uint idx = 0;
    
#endif
#ifdef USE_CUDA
  dim3 grid,threads;
#endif
  int size_mpi_buffer;

  //get mesh pointer out of Fortran integer container
  Mesh *mp = (Mesh *) *Mesh_pointer_f;

  // crust/mantle region
  if (*IREGION == IREGION_CRUST_MANTLE) {
    size_mpi_buffer = NDIM*mp->max_nibool_interfaces_cm*mp->num_interfaces_crust_mantle;

    if( size_mpi_buffer > 0 ){

      blocksize = BLOCKSIZE_TRANSFER;
      size_padded = ((int) ceil (((double) mp->max_nibool_interfaces_cm) / ((double) blocksize))) * blocksize;

      get_blocks_xy (size_padded / blocksize, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
      if (run_opencl) {
        cl_event kernel_evt;
        
        local_work_size[0] = blocksize;
        local_work_size[1] = 1;
        global_work_size[0] = num_blocks_x * blocksize;
        global_work_size[1] = num_blocks_y;

        if (*FORWARD_OR_ADJOINT == 1) {
            
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_accel_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_send_accel_buffer_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->num_interfaces_crust_mantle));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->max_nibool_interfaces_cm));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_nibool_interfaces_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_ibool_interfaces_crust_mantle.ocl));

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.prepare_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, &kernel_evt));

          // copies buffer to CPU
          if (GPU_ASYNC_COPY) {
            if (mp->has_last_copy_evt) {
              clCheck (clReleaseEvent (mp->last_copy_evt));
            }
            clCheck (clEnqueueReadBuffer (mocl.copy_queue, mp->d_send_accel_buffer_crust_mantle.ocl, CL_FALSE, 0,
                                          size_mpi_buffer * sizeof (realw),
                                          mp->h_send_accel_buffer_cm, 1, &kernel_evt, &mp->last_copy_evt));
            mp->has_last_copy_evt = 1;
          } else {
            clCheck (clEnqueueReadBuffer (mocl.command_queue, mp->d_send_accel_buffer_crust_mantle.ocl, CL_TRUE, 0,
                                          size_mpi_buffer * sizeof (realw),
                                          send_accel_buffer, 0, NULL, NULL));
          }
        } else if (*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY ();

          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_b_accel_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_b_send_accel_buffer_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->num_interfaces_crust_mantle));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->max_nibool_interfaces_cm));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_nibool_interfaces_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_ibool_interfaces_crust_mantle.ocl));

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.prepare_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, &kernel_evt));

          // copies buffer to CPU
          if (GPU_ASYNC_COPY) {
            if (mp->has_last_copy_evt) {
              clCheck (clReleaseEvent (mp->last_copy_evt));
            }
            
            clCheck (clEnqueueReadBuffer (mocl.copy_queue, mp->d_b_send_accel_buffer_crust_mantle.ocl, CL_TRUE, 0,
                                          size_mpi_buffer * sizeof (realw),
                                          mp->h_b_send_accel_buffer_cm, 1, &kernel_evt, &mp->last_copy_evt));
            mp->has_last_copy_evt = 1;
          } else {
            clCheck (clEnqueueReadBuffer (mocl.command_queue, mp->d_send_accel_buffer_crust_mantle.ocl, CL_TRUE, 0,
                                          size_mpi_buffer * sizeof (realw),
                                          send_accel_buffer, 0, NULL, NULL));
          }
        }
        clReleaseEvent (kernel_evt);
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        grid = dim3(num_blocks_x,num_blocks_y);
        threads = dim3(blocksize,1,1);

        if(*FORWARD_OR_ADJOINT == 1) {
          prepare_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_accel_crust_mantle.cuda,
                                                             mp->d_send_accel_buffer_crust_mantle.cuda,
                                                             mp->num_interfaces_crust_mantle,
                                                             mp->max_nibool_interfaces_cm,
                                                             mp->d_nibool_interfaces_crust_mantle.cuda,
                                                             mp->d_ibool_interfaces_crust_mantle.cuda);

          // copies buffer to CPU
          if( GPU_ASYNC_COPY ){
            // waits until kernel is finished before starting async memcpy
            cudaStreamSynchronize(mp->compute_stream);
            // copies buffer to CPU
            cudaMemcpyAsync(mp->h_send_accel_buffer_cm,mp->d_send_accel_buffer_crust_mantle.cuda,size_mpi_buffer*sizeof(realw),
                            cudaMemcpyDeviceToHost,mp->copy_stream);
          }else{
            // synchronous copy
            print_CUDA_error_if_any(cudaMemcpy(send_accel_buffer,mp->d_send_accel_buffer_crust_mantle.cuda,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyDeviceToHost),41000);
          }
        } else if(*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY();

          prepare_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_b_accel_crust_mantle.cuda,
                                                             mp->d_b_send_accel_buffer_crust_mantle.cuda,
                                                             mp->num_interfaces_crust_mantle,
                                                             mp->max_nibool_interfaces_cm,
                                                             mp->d_nibool_interfaces_crust_mantle.cuda,
                                                             mp->d_ibool_interfaces_crust_mantle.cuda);
          // copies buffer to CPU
          if (GPU_ASYNC_COPY) {
            // waits until kernel is finished before starting async memcpy
            cudaStreamSynchronize(mp->compute_stream);
            // copies buffer to CPU
            cudaMemcpyAsync(mp->h_b_send_accel_buffer_cm,mp->d_b_send_accel_buffer_crust_mantle.cuda,size_mpi_buffer*sizeof(realw),
                            cudaMemcpyDeviceToHost,mp->copy_stream);
          } else {
            // synchronous copy
            print_CUDA_error_if_any(cudaMemcpy(send_accel_buffer,mp->d_b_send_accel_buffer_crust_mantle.cuda,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyDeviceToHost),41001);
            
          }
        }
      }
#endif
    }
  }

  // inner core region
  if (*IREGION == IREGION_INNER_CORE) {
    size_mpi_buffer = NDIM*mp->max_nibool_interfaces_ic*mp->num_interfaces_inner_core;

    if( size_mpi_buffer > 0 ){

      blocksize = BLOCKSIZE_TRANSFER;
      size_padded = ((int) ceil (((double) mp->max_nibool_interfaces_ic) / ((double) blocksize))) * blocksize;

      get_blocks_xy (size_padded / blocksize, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
      if (run_opencl) {
        cl_event kernel_evt;
        
        idx = 0;
        
        if (*FORWARD_OR_ADJOINT == 1) {
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_accel_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_send_accel_buffer_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->num_interfaces_inner_core));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->max_nibool_interfaces_ic));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_nibool_interfaces_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_ibool_interfaces_inner_core.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.prepare_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, &kernel_evt));

          // copies buffer to CPU
          if (GPU_ASYNC_COPY) {
            if (mp->has_last_copy_evt) {
              clCheck (clReleaseEvent (mp->last_copy_evt));
            }
            
            clCheck (clEnqueueReadBuffer (mocl.copy_queue, mp->d_send_accel_buffer_inner_core.ocl, CL_TRUE, 0,
                                          size_mpi_buffer * sizeof (realw),
                                          mp->h_send_accel_buffer_ic, 1, &kernel_evt, &mp->last_copy_evt));
            mp->has_last_copy_evt = 1;
          } else {
            clCheck (clEnqueueReadBuffer (mocl.command_queue, mp->d_send_accel_buffer_inner_core.ocl, CL_TRUE, 0,
                                          size_mpi_buffer * sizeof (realw),
                                          send_accel_buffer, 0, NULL, NULL));
          }

        } else if (*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY ();

          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_b_accel_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_b_send_accel_buffer_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->num_interfaces_inner_core));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->max_nibool_interfaces_ic));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_nibool_interfaces_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.prepare_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_ibool_interfaces_inner_core.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.prepare_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, &kernel_evt));

          // copies buffer to CPU
          if (GPU_ASYNC_COPY) {
            if (mp->has_last_copy_evt) {
              clCheck (clReleaseEvent (mp->last_copy_evt));
            }
            
            clCheck (clEnqueueReadBuffer (mocl.copy_queue, mp->d_b_send_accel_buffer_inner_core.ocl, CL_FALSE, 0,
                                          size_mpi_buffer * sizeof (realw),
                                          mp->h_b_send_accel_buffer_ic, 1, &kernel_evt, &mp->last_copy_evt));
            mp->has_last_copy_evt = 1;
          } else {            
            clCheck (clEnqueueReadBuffer (mocl.command_queue, mp->d_b_send_accel_buffer_inner_core.ocl, CL_TRUE, 0,
                                          size_mpi_buffer * sizeof (realw),
                                          send_accel_buffer, 0, NULL, NULL));
          }
        }
        clReleaseEvent (kernel_evt);
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        grid = dim3(num_blocks_x,num_blocks_y);
        threads = dim3(blocksize,1,1);

        if(*FORWARD_OR_ADJOINT == 1) {
          prepare_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_accel_inner_core.cuda,
                                                                                  mp->d_send_accel_buffer_inner_core.cuda,
                                                                                  mp->num_interfaces_inner_core,
                                                                                  mp->max_nibool_interfaces_ic,
                                                                                  mp->d_nibool_interfaces_inner_core.cuda,
                                                                                  mp->d_ibool_interfaces_inner_core.cuda);

          // copies buffer to CPU
          if( GPU_ASYNC_COPY ){
            // waits until kernel is finished before starting async memcpy
            cudaStreamSynchronize(mp->compute_stream);
            // copies buffer to CPU
            cudaMemcpyAsync(mp->h_send_accel_buffer_ic,mp->d_send_accel_buffer_inner_core.cuda,size_mpi_buffer*sizeof(realw),
                            cudaMemcpyDeviceToHost,mp->copy_stream);
          }else{
            // synchronous copy
            print_CUDA_error_if_any(cudaMemcpy(send_accel_buffer,mp->d_send_accel_buffer_inner_core.cuda,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyDeviceToHost),41000);
            
          }
        } else if (*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY();
          
          prepare_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_b_accel_inner_core.cuda,
                                                                                  mp->d_b_send_accel_buffer_inner_core.cuda,
                                                                                  mp->num_interfaces_inner_core,
                                                                                  mp->max_nibool_interfaces_ic,
                                                                                  mp->d_nibool_interfaces_inner_core.cuda,
                                                                                  mp->d_ibool_interfaces_inner_core.cuda);
          // copies buffer to CPU
          if( GPU_ASYNC_COPY ){
            // waits until kernel is finished before starting async memcpy
            cudaStreamSynchronize(mp->compute_stream);
            // copies buffer to CPU
            cudaMemcpyAsync(mp->h_b_send_accel_buffer_ic,mp->d_b_send_accel_buffer_inner_core.cuda,size_mpi_buffer*sizeof(realw),
                            cudaMemcpyDeviceToHost,mp->copy_stream);
          }else{
            // synchronous copy
            print_CUDA_error_if_any(cudaMemcpy(send_accel_buffer,mp->d_b_send_accel_buffer_inner_core.cuda,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyDeviceToHost),41001);
          }
        }
      }
#endif
    }
  }
#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("transfer_boun_from_device");
#endif
}

// FORWARD_OR_ADJOINT == 1 for accel, and == 3 for b_accel
extern EXTERN_LANG
void FC_FUNC_ (transfer_asmbl_accel_to_device,
               TRANSFER_ASMBL_ACCEL_TO_DEVICE) (long *Mesh_pointer,
                                                realw *buffer_recv_vector,
                                                int *IREGION,
                                                int *FORWARD_OR_ADJOINT) {
  TRACE ("transfer_asmbl_accel_to_device");

  int blocksize, size_padded;
  int num_blocks_x, num_blocks_y;
  int size_mpi_buffer;
  
#ifdef USE_OPENCL
  size_t global_work_size[2];
  size_t local_work_size[2];
  cl_uint idx = 0;
  cl_event *copy_evt = NULL;
  cl_uint num_evt = 0;
#endif
#ifdef USE_CUDA
  dim3 grid,threads;
#endif

  Mesh *mp = (Mesh *) *Mesh_pointer;     //get mesh pointer out of Fortran integer container

  // crust/mantle region
  if (*IREGION == IREGION_CRUST_MANTLE) {
    size_mpi_buffer = NDIM*(mp->max_nibool_interfaces_cm)*(mp->num_interfaces_crust_mantle);

    if( size_mpi_buffer > 0 ){
      // assembles values
      blocksize = BLOCKSIZE_TRANSFER;
      size_padded = ((int) ceil (((double) mp->max_nibool_interfaces_cm) / ((double) blocksize))) * blocksize;

      get_blocks_xy (size_padded / blocksize, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
      if (run_opencl) {        
        if (GPU_ASYNC_COPY && mp->has_last_copy_evt) {
          copy_evt = &mp->last_copy_evt;
          num_evt = 1;
        }
        
        if (*FORWARD_OR_ADJOINT == 1) {
          // copies vector buffer values to GPU
          if (!GPU_ASYNC_COPY) {
            clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_send_accel_buffer_crust_mantle.ocl, CL_FALSE, 0,
                                           NDIM * (mp->max_nibool_interfaces_cm) * (mp->num_interfaces_crust_mantle)*sizeof (realw),
                                           buffer_recv_vector, 0, NULL, NULL));
          }
          //assemble forward accel
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_accel_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_send_accel_buffer_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->num_interfaces_crust_mantle));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->max_nibool_interfaces_cm));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_nibool_interfaces_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_ibool_interfaces_crust_mantle.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.assemble_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, num_evt, copy_evt, NULL));
        } else if (*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY ();
          
          if (!GPU_ASYNC_COPY) {
            // copies vector buffer values to GPU
            clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_b_send_accel_buffer_crust_mantle.ocl, CL_FALSE, 0,
                                           NDIM * (mp->max_nibool_interfaces_cm) * (mp->num_interfaces_crust_mantle) * sizeof (realw),
                                           buffer_recv_vector, 0, NULL, NULL));
          }
          
          //assemble adjoint accel
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_b_accel_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_b_send_accel_buffer_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->num_interfaces_crust_mantle));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->max_nibool_interfaces_cm));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_nibool_interfaces_crust_mantle.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_ibool_interfaces_crust_mantle.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.assemble_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, num_evt, copy_evt, NULL));
        }
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        grid = dim3(num_blocks_x,num_blocks_y);
        threads = dim3(blocksize,1,1);

        if(*FORWARD_OR_ADJOINT == 1) {
          // asynchronous copy
          if( GPU_ASYNC_COPY ){
            // Wait until previous copy stream finishes. We assemble while other compute kernels execute.
            cudaStreamSynchronize(mp->copy_stream);
          }else{
            // (cudaMemcpy implicitly synchronizes all other cuda operations)
            // copies vector buffer values to GPU
            print_CUDA_error_if_any(cudaMemcpy(mp->d_send_accel_buffer_crust_mantle.cuda,buffer_recv_vector,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyHostToDevice),41000);
          }

          //assemble forward accel
          assemble_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_accel_crust_mantle.cuda,
                                                              mp->d_send_accel_buffer_crust_mantle.cuda,
                                                              mp->num_interfaces_crust_mantle,
                                                              mp->max_nibool_interfaces_cm,
                                                              mp->d_nibool_interfaces_crust_mantle.cuda,
                                                              mp->d_ibool_interfaces_crust_mantle.cuda);
        }
        else if(*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY();

          // asynchronous copy
          if( GPU_ASYNC_COPY ){
            // Wait until previous copy stream finishes. We assemble while other compute kernels execute.
            cudaStreamSynchronize(mp->copy_stream);
          }else{
            // (cudaMemcpy implicitly synchronizes all other cuda operations)
            // copies vector buffer values to GPU
            print_CUDA_error_if_any(cudaMemcpy(mp->d_b_send_accel_buffer_crust_mantle.cuda, buffer_recv_vector,
                                               NDIM*(mp->max_nibool_interfaces_cm)*(mp->num_interfaces_crust_mantle)*sizeof(realw),
                                               cudaMemcpyHostToDevice),41000);
          }

          //assemble adjoint accel
          assemble_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_b_accel_crust_mantle.cuda,
                                                              mp->d_b_send_accel_buffer_crust_mantle.cuda,
                                                              mp->num_interfaces_crust_mantle,
                                                              mp->max_nibool_interfaces_cm,
                                                              mp->d_nibool_interfaces_crust_mantle.cuda,
                                                              mp->d_ibool_interfaces_crust_mantle.cuda);
        }
      }
#endif
    }
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("transfer_asmbl_accel_to_device in crust_mantle");
#endif

  // inner core region
  if (*IREGION == IREGION_INNER_CORE) {
    size_mpi_buffer = NDIM*(mp->max_nibool_interfaces_ic)*(mp->num_interfaces_inner_core);

    if (size_mpi_buffer > 0) {
      // assembles values
      blocksize = BLOCKSIZE_TRANSFER;
      size_padded = ((int) ceil (((double) mp->max_nibool_interfaces_ic) / ((double) blocksize))) * blocksize;

      get_blocks_xy (size_padded / blocksize, &num_blocks_x, &num_blocks_y);

#ifdef USE_OPENCL
      if (run_opencl) {
        idx = 0;

        if (GPU_ASYNC_COPY && mp->has_last_copy_evt) {
          copy_evt = &mp->last_copy_evt;
          num_evt = 1;
        }
        
        if (*FORWARD_OR_ADJOINT == 1) {
          // copies buffer values to GPU
          if (!GPU_ASYNC_COPY) {
            clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_send_accel_buffer_inner_core.ocl, CL_FALSE, 0,
                                           NDIM * (mp->max_nibool_interfaces_ic) * (mp->num_interfaces_inner_core)*sizeof (realw),
                                           buffer_recv_vector, 0, NULL, NULL));
          }
          //assemble forward accel
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_accel_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_send_accel_buffer_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->num_interfaces_inner_core));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->max_nibool_interfaces_ic));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_nibool_interfaces_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_ibool_interfaces_inner_core.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.assemble_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
        } else if (*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY ();
          
          if (!GPU_ASYNC_COPY) {
            // copies buffer values to GPU
            clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_b_send_accel_buffer_inner_core.ocl, CL_FALSE, 0,
                                           NDIM * (mp->max_nibool_interfaces_ic) * (mp->num_interfaces_inner_core) * sizeof (realw),
                                           buffer_recv_vector, 0, NULL, NULL));
          }
          //assemble adjoint accel
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_b_accel_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_b_send_accel_buffer_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->num_interfaces_inner_core));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (int), (void *) &mp->max_nibool_interfaces_ic));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_nibool_interfaces_inner_core.ocl));
          clCheck (clSetKernelArg (mocl.kernels.assemble_boundary_accel_on_device, idx++, sizeof (cl_mem), (void *) &mp->d_ibool_interfaces_inner_core.ocl));

          local_work_size[0] = blocksize;
          local_work_size[1] = 1;
          global_work_size[0] = num_blocks_x * blocksize;
          global_work_size[1] = num_blocks_y;

          clCheck (clEnqueueNDRangeKernel (mocl.command_queue, mocl.kernels.assemble_boundary_accel_on_device, 2, NULL, global_work_size, local_work_size, num_evt, copy_evt, NULL));
        }

        if (GPU_ASYNC_COPY && mp->has_last_copy_evt) {
          clCheck (clReleaseEvent (mp->last_copy_evt));
          mp->has_last_copy_evt = 0;
        }
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        grid = dim3(num_blocks_x,num_blocks_y);
        threads = dim3(blocksize,1,1);

        if(*FORWARD_OR_ADJOINT == 1) {
          if( GPU_ASYNC_COPY ){
            // Wait until previous copy stream finishes. We assemble while other compute kernels execute.
            cudaStreamSynchronize(mp->copy_stream);
          }else{
            // (cudaMemcpy implicitly synchronizes all other cuda operations)
            // copies buffer values to GPU
            print_CUDA_error_if_any(cudaMemcpy(mp->d_send_accel_buffer_inner_core.cuda,buffer_recv_vector,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyHostToDevice),41001);
          }

          //assemble forward accel
          assemble_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_accel_inner_core.cuda,
                                                              mp->d_send_accel_buffer_inner_core.cuda,
                                                              mp->num_interfaces_inner_core,
                                                              mp->max_nibool_interfaces_ic,
                                                              mp->d_nibool_interfaces_inner_core.cuda,
                                                              mp->d_ibool_interfaces_inner_core.cuda);
        }
        else if(*FORWARD_OR_ADJOINT == 3) {
          // debug
          DEBUG_BACKWARD_ASSEMBLY();

          if (GPU_ASYNC_COPY ){
            // Wait until previous copy stream finishes. We assemble while other compute kernels execute.
            cudaStreamSynchronize(mp->copy_stream);
          }else{
            // (cudaMemcpy implicitly synchronizes all other cuda operations)
            // copies buffer values to GPU
            print_CUDA_error_if_any(cudaMemcpy(mp->d_b_send_accel_buffer_inner_core.cuda,buffer_recv_vector,size_mpi_buffer*sizeof(realw),
                                               cudaMemcpyHostToDevice),41001);
          }

          //assemble adjoint accel
          assemble_boundary_accel_on_device<<<grid,threads,0,mp->compute_stream>>>(mp->d_b_accel_inner_core.cuda,
                                                              mp->d_b_send_accel_buffer_inner_core.cuda,
                                                              mp->num_interfaces_inner_core,
                                                              mp->max_nibool_interfaces_ic,
                                                              mp->d_nibool_interfaces_inner_core.cuda,
                                                              mp->d_ibool_interfaces_inner_core.cuda);
        }
      }
#endif
    }
  }

#ifdef ENABLE_VERY_SLOW_ERROR_CHECKING
  exit_on_gpu_error ("transfer_asmbl_accel_to_device in inner_core");
#endif
}


/* ----------------------------------------------------------------------------------------------- */
// Asynchronous memory copy for MPI buffers
/* ----------------------------------------------------------------------------------------------- */

extern EXTERN_LANG
void FC_FUNC_(transfer_buffer_to_device_async,
              TRANSFER_BUFFER_TO_DEVICE_ASYNC)(long* Mesh_pointer,
                                             realw* buffer,
                                             int* IREGION,
                                             int* FORWARD_OR_ADJOINT)
{
  // asynchronous transfer from host to device
  TRACE("transfer_buffer_to_device_async");

  int size_mpi_buffer;
  Mesh *mp = (Mesh *)(*Mesh_pointer); // get Mesh from Fortran integer wrapper

  // checks async-memcpy
  if (GPU_ASYNC_COPY == 0) {
    exit_on_error("transfer_buffer_to_device_async must be called with GPU_ASYNC_COPY == 1, please check mesh_constants_cuda.h");
  }

  // regions
  if (*IREGION == IREGION_CRUST_MANTLE) {
    // crust/mantle region
    size_mpi_buffer = NDIM * mp->max_nibool_interfaces_cm * mp->num_interfaces_crust_mantle;

    if (size_mpi_buffer > 0) {

      if (*FORWARD_OR_ADJOINT == 1) {
        // copy on host memory
        memcpy(mp->h_recv_accel_buffer_cm, buffer, size_mpi_buffer * sizeof(realw));

        // asynchronous copy to GPU using copy_stream
#ifdef USE_OPENCL
        if (run_opencl) {
          clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_send_accel_buffer_crust_mantle.ocl, CL_FALSE, 0,
                                         size_mpi_buffer * sizeof (realw),
                                         mp->h_recv_accel_buffer_cm, 0, NULL, NULL));
        }
#endif
#ifdef USE_CUDA
        if (run_cuda) {
          cudaMemcpyAsync(mp->d_send_accel_buffer_crust_mantle.cuda, mp->h_recv_accel_buffer_cm,size_mpi_buffer*sizeof(realw),
                          cudaMemcpyHostToDevice,mp->copy_stream);
        }
#endif

      } else if (*FORWARD_OR_ADJOINT == 3) {
        // debug
        DEBUG_BACKWARD_ASSEMBLY();

        // copy on host memory
        memcpy (mp->h_b_recv_accel_buffer_cm, buffer, size_mpi_buffer * sizeof(realw));

        // asynchronous copy to GPU using copy_stream
#ifdef USE_OPENCL
        if (run_opencl) {
          clCheck (clEnqueueWriteBuffer (mocl.command_queue, mp->d_b_send_accel_buffer_crust_mantle.ocl, CL_FALSE, 0,
                                         size_mpi_buffer * sizeof (realw),
                                         mp->h_b_recv_accel_buffer_cm, 0, NULL, NULL));
        }
#endif    
#ifdef USE_CUDA
        if (run_cuda) {
          cudaMemcpyAsync(mp->d_b_send_accel_buffer_crust_mantle.cuda,mp->h_b_recv_accel_buffer_cm,size_mpi_buffer*sizeof(realw),
                        cudaMemcpyHostToDevice,mp->copy_stream);
        }
#endif
      }
    }
  } else if (*IREGION == IREGION_INNER_CORE) {
    // inner core region
    size_mpi_buffer = NDIM * mp->max_nibool_interfaces_ic * mp->num_interfaces_inner_core;

    if (size_mpi_buffer > 0) {

      if (*FORWARD_OR_ADJOINT == 1) {
        // copy on host memory
        memcpy(mp->h_recv_accel_buffer_ic, buffer, size_mpi_buffer * sizeof(realw));

        // asynchronous copy to GPU using copy_stream
#ifdef USE_OPENCL
        if (run_opencl) {
          clCheck(clEnqueueWriteBuffer(mocl.command_queue, mp->d_send_accel_buffer_inner_core.ocl, CL_FALSE, 0,
                                         size_mpi_buffer * sizeof (realw),
                                         mp->h_recv_accel_buffer_ic, 0, NULL, NULL));
        }
#endif
#ifdef USE_CUDA
        if (run_cuda) {
          cudaMemcpyAsync(mp->d_send_accel_buffer_inner_core.cuda,mp->h_recv_accel_buffer_ic,size_mpi_buffer*sizeof(realw),
                        cudaMemcpyHostToDevice,mp->copy_stream);
        }
#endif

      } else if (*FORWARD_OR_ADJOINT == 3) {
        // debug
        DEBUG_BACKWARD_ASSEMBLY();

        // copy on host memory
        memcpy(mp->h_b_recv_accel_buffer_ic, buffer, size_mpi_buffer * sizeof(realw));

        // asynchronous copy to GPU using copy_stream
#ifdef USE_OPENCL
        if (run_opencl) {
          clCheck(clEnqueueWriteBuffer(mocl.command_queue, mp->d_b_send_accel_buffer_inner_core.ocl, CL_FALSE, 0,
                                       size_mpi_buffer * sizeof (realw),
                                       mp->h_b_recv_accel_buffer_ic, 0, NULL, NULL));
        }
#endif
#ifdef USE_CUDA
        if (run_cuda) {
          cudaMemcpyAsync(mp->d_b_send_accel_buffer_inner_core.cuda, mp->h_b_recv_accel_buffer_ic,size_mpi_buffer*sizeof(realw),
                        cudaMemcpyHostToDevice,mp->copy_stream);
        }
#endif
      }
    }
  } else if (*IREGION == IREGION_OUTER_CORE) {
    // outer core region
    size_mpi_buffer = mp->max_nibool_interfaces_oc * mp->num_interfaces_outer_core;

    if (size_mpi_buffer > 0) {

      if (*FORWARD_OR_ADJOINT == 1) {
        // copy on host memory
        memcpy(mp->h_recv_accel_buffer_oc, buffer, size_mpi_buffer * sizeof(realw));

        // asynchronous copy to GPU using copy_stream
#ifdef USE_OPENCL
        if (run_opencl) {
          clCheck(clEnqueueWriteBuffer(mocl.command_queue, mp->d_send_accel_buffer_outer_core.ocl, CL_FALSE, 0,
                                       size_mpi_buffer * sizeof (realw),
                                       mp->h_recv_accel_buffer_oc, 0, NULL, NULL));
      }
#endif
#ifdef USE_CUDA
        if (run_cuda) {
          cudaMemcpyAsync(mp->d_send_accel_buffer_outer_core.cuda, mp->h_recv_accel_buffer_oc,size_mpi_buffer*sizeof(realw),
                        cudaMemcpyHostToDevice,mp->copy_stream);
        }
#endif

      } else if (*FORWARD_OR_ADJOINT == 3) {
        // debug
        DEBUG_BACKWARD_ASSEMBLY();

        // copy on host memory
        memcpy(mp->h_b_recv_accel_buffer_oc, buffer, size_mpi_buffer * sizeof(realw));

        // asynchronous copy to GPU using copy_stream
#ifdef USE_OPENCL
        if (run_opencl) {
          clCheck(clEnqueueWriteBuffer(mocl.command_queue, mp->d_b_send_accel_buffer_outer_core.ocl, CL_FALSE, 0,
                                       size_mpi_buffer * sizeof (realw),
                                       mp->h_b_recv_accel_buffer_oc, 0, NULL, NULL));
      }
#endif
#ifdef USE_CUDA
        if (run_cuda) {
          cudaMemcpyAsync(mp->d_b_send_accel_buffer_outer_core.cuda, mp->h_b_recv_accel_buffer_oc,size_mpi_buffer*sizeof(realw),
                        cudaMemcpyHostToDevice,mp->copy_stream);
        }
#endif
      }
    }
  }
}


/* ----------------------------------------------------------------------------------------------- */

extern EXTERN_LANG
void FC_FUNC_(sync_copy_from_device,
              SYNC_copy_FROM_DEVICE)(long* Mesh_pointer,
                                     int* iphase,
                                     realw* send_buffer,
                                     int* IREGION,
                                     int* FORWARD_OR_ADJOINT)
{
  // synchronizes copy stream before copying buffers from pinned memory to CPU host
  TRACE("sync_copy_from_device");

  int size_mpi_buffer;
  Mesh *mp = (Mesh *)(*Mesh_pointer); // get Mesh from Fortran integer wrapper

  // checks async-memcpy
  if (GPU_ASYNC_COPY == 0) {
    exit_on_error("sync_copy_from_device must be called with GPU_ASYNC_COPY == 1, please check mesh_constants_gpu.h");
  }

  // Wait until async-memcpy of outer elements is finished and start MPI.
  if (*iphase != 2) {
    exit_on_error("sync_copy_from_device must be called for iphase == 2");
  }
  
  // regions
  if (*IREGION == IREGION_CRUST_MANTLE) {
    // crust/mantle
    size_mpi_buffer = NDIM * mp->max_nibool_interfaces_cm * mp->num_interfaces_crust_mantle;

    if (size_mpi_buffer > 0) {
      // waits for asynchronous copy to finish
#ifdef USE_OPENCL
      if (run_opencl) {
        if (mp->has_last_copy_evt) {
          clCheck (clReleaseEvent (mp->last_copy_evt));
          mp->has_last_copy_evt = 0;
        }
        
        clCheck (clFinish (mocl.copy_queue));
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        cudaStreamSynchronize(mp->copy_stream);
      }
#endif
      if (*FORWARD_OR_ADJOINT == 1) {
        // There have been problems using the pinned-memory with MPI, so
        // we copy the buffer into a non-pinned region.
        memcpy(send_buffer, mp->h_send_accel_buffer_cm, size_mpi_buffer * sizeof(realw));

      } else if (*FORWARD_OR_ADJOINT == 3) {
        // we copy the buffer into a non-pinned region.
        memcpy(send_buffer, mp->h_b_send_accel_buffer_cm, size_mpi_buffer * sizeof(realw));
      }
    }
  } else if (*IREGION == IREGION_INNER_CORE) {
    // inner core
    size_mpi_buffer = NDIM * mp->max_nibool_interfaces_ic * mp->num_interfaces_inner_core;

    if (size_mpi_buffer > 0) {
      // waits for asynchronous copy to finish
#ifdef USE_OPENCL
      if (run_opencl) {
        if (mp->has_last_copy_evt) {
          clCheck (clReleaseEvent (mp->last_copy_evt));
          mp->has_last_copy_evt = 0;
        }
        
        clCheck (clFinish (mocl.copy_queue));
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        cudaStreamSynchronize(mp->copy_stream);
      }
#endif

      if (*FORWARD_OR_ADJOINT == 1) {
        // There have been problems using the pinned-memory with MPI, so
        // we copy the buffer into a non-pinned region.
        memcpy(send_buffer,mp->h_send_accel_buffer_ic,size_mpi_buffer*sizeof(realw));

      } else if(*FORWARD_OR_ADJOINT == 3) {
        // we copy the buffer into a non-pinned region.
        memcpy(send_buffer,mp->h_b_send_accel_buffer_ic,size_mpi_buffer*sizeof(realw));
      }
    }
  } else if (*IREGION == IREGION_OUTER_CORE) {
    // outer core
    size_mpi_buffer = mp->max_nibool_interfaces_oc * mp->num_interfaces_outer_core;

    if (size_mpi_buffer > 0) {
      // waits for asynchronous copy to finish
#ifdef USE_OPENCL
      if (run_opencl) {
        if (mp->has_last_copy_evt) {
          clCheck (clReleaseEvent (mp->last_copy_evt));
          mp->has_last_copy_evt = 0;
        }
        
        clCheck (clFinish (mocl.copy_queue));
      }
#endif
#ifdef USE_CUDA
      if (run_cuda) {
        cudaStreamSynchronize(mp->copy_stream);
      }
#endif

      if (*FORWARD_OR_ADJOINT == 1) {
        // There have been problems using the pinned-memory with MPI, so
        // we copy the buffer into a non-pinned region.
        memcpy(send_buffer, mp->h_send_accel_buffer_oc, size_mpi_buffer * sizeof(realw));

      } else if (*FORWARD_OR_ADJOINT == 3) {
        // we copy the buffer into a non-pinned region.
        memcpy(send_buffer, mp->h_b_send_accel_buffer_oc, size_mpi_buffer * sizeof(realw));
      }
    }
  }
  // memory copy is now finished, so non-blocking MPI send can proceed
}

