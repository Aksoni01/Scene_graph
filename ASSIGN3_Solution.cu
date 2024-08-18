
/*
	CS 6023 Assignment 3.
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
using namespace std;

void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;


	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ;
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ;
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}

	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


__global__ void kerneltranslation(int *gpu_hOffset,int *gpu_hCsr,int *gpu_meshID,int *gpu_commandID,int *gpu_movementID,int *gpu_hGlobalCoordinatesX,int *gpu_hGlobalCoordinatesY,int nums,int *gpu_postorder,int *gpu_post_size,int *gpu_post_offset){
		 int id= threadIdx.x + blockDim.x*blockIdx.x;
		 if(id<nums){
            
				    int meshnumber=gpu_meshID[id],command=gpu_commandID[id],movement=gpu_movementID[id];
            /* iterate for postorder of that node */
            int start = gpu_post_offset[meshnumber];
            int end= start - gpu_post_size[gpu_post_offset[meshnumber]] ;
      
            for(int j=start;j>end;j--){
                if(command==0){
                    atomicSub(&gpu_hGlobalCoordinatesX[gpu_postorder[j]],movement);
                }
                else if(command==1){
                    atomicAdd(&gpu_hGlobalCoordinatesX[gpu_postorder[j]],movement);
                }
                else if(command==2){
                    atomicSub(&gpu_hGlobalCoordinatesY[gpu_postorder[j]],movement);
                }
                else {
                    atomicAdd(&gpu_hGlobalCoordinatesY[gpu_postorder[j]],movement);
                }
            }
          }
}
__host__ int traversal(int *hOffset,int *Csr,int node,int *id ,vector<pair<int,int>> &post_size,int *offset){
    int subtreeSize = 1; // Initialize the subtree size to 1 for the current node

    for (int i = hOffset[node]; i < hOffset[node + 1]; ++i) {
        // Recursive call for child nodes
        subtreeSize += traversal(hOffset, Csr, Csr[i], id, post_size,offset);
    }
    
    post_size.push_back({node,subtreeSize});
    offset[node]=*id;
    // Increment the index for the next node
    ++*id;
    return subtreeSize;
}



__global__ void storing_Opacity(int *result,int *gpu_hOpacity,int *gpu_hFrameSizeX,int *gpu_hFrameSizeY,int *gpu_hGlobalCoordinatesX,int *gpu_hGlobalCoordinatesY,int frameSizeX,int frameSizeY,int nums,int *maxi_opacity){
      int id = threadIdx.x + blockDim.x * blockIdx.x;
      if(id<nums){
        atomicMax(maxi_opacity,gpu_hOpacity[id]);

        int  startX = gpu_hGlobalCoordinatesX[id] ,startY= gpu_hGlobalCoordinatesY[id];
        int endX=  gpu_hGlobalCoordinatesX[id] + gpu_hFrameSizeX[id],endY = gpu_hGlobalCoordinatesY[id] + gpu_hFrameSizeY[id];
        for(int i= startX;i<endX;i++){
          for(int j=startY;j<endY;j++){
            if(i>=0 && i<frameSizeX && j>=0 && j<frameSizeY){
                atomicMax(&result[i*frameSizeY+j],gpu_hOpacity[id]);}
          }
        }
      }
}

__global__ void make_opacity_map(int *opacity_mapping,int *gpu_hOpacity,int nums){
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id<nums){
    opacity_mapping[gpu_hOpacity[id]]=id;
  }
}

__global__ void resultantScene(int *result,int **gpu_hMesh,int *opacity_mapping,int *gpu_hGlobalCoordinatesX,int *gpu_hGlobalCoordinatesY,int *gpu_frameSizeX,int *gpu_frameSizeY,int frameSizeX,int frameSizeY){
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id<frameSizeX * frameSizeY){
     int row=id/frameSizeY,col=id%frameSizeY;
     if(result[id]==-1){
      result[id]=0;
     }
     else{
         int mesh_temp=opacity_mapping[result[id]];
        result[id]=gpu_hMesh[mesh_temp][(row-gpu_hGlobalCoordinatesX[mesh_temp])*gpu_frameSizeY[mesh_temp] + (col-gpu_hGlobalCoordinatesY[mesh_temp])];
        }
  }
}

int main (int argc, char **argv) {

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ;

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;

	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;

	// Code begins here.
	// Do not change anything above this comment.


  vector<pair<int,int>> post_size;
 int *offset,id=0,source=0;
 offset=(int *) malloc(sizeof(int)*V);
 int dy=traversal(hOffset,hCsr,source,&id,post_size,offset);
 int *postorder,*postorder_size;
 postorder=(int *)malloc(sizeof(int) * V);
 postorder_size=(int *)malloc(sizeof(int)*V);
 for(int i=0;i<post_size.size();i++){
   postorder[i]=post_size[i].first;
   postorder_size[i]=post_size[i].second;
 }


 int *meshID,*commandID,*movementId;
 meshID=(int*)malloc(sizeof(int)*numTranslations);
 commandID=(int*)malloc(sizeof(int)*numTranslations);
 movementId=(int*)malloc(sizeof(int)*numTranslations);

 for(int i=0; i<numTranslations; i++) {
			meshID[i]=translations[i][0];
      commandID[i]=translations[i][1];
      movementId[i]=translations[i][2];
	}

  int *gpu_hOffset, *gpu_hCsr , *gpu_hOpacity, *gpu_hGlobalCoordinatesX,*gpu_hGlobalCoordinatesY,*gpu_hFrameSizeX,*gpu_hFrameSizeY ,*gpu_meshID,*gpu_commandID,*gpu_movementID,*result,*gpu_postorder,*gpu_post_size,*gpu_post_offset;
  int **gpu_hMesh;
  int **dummy;
  
  dummy=(int **)malloc(sizeof(int *)*V);
	cudaMalloc(&gpu_hMesh, sizeof(int*) * V);
	for (int i = 0; i < V; ++i) {
			int size = hFrameSizeX[i] * hFrameSizeY[i];
			int *tempMesh;
			cudaMalloc(&tempMesh, sizeof(int) * size);
			cudaMemcpy(tempMesh, hMesh[i], sizeof(int) * size, cudaMemcpyHostToDevice);
			dummy[i]=tempMesh;
	}
  cudaMemcpy(gpu_hMesh,dummy, sizeof(int*) * (V) , cudaMemcpyHostToDevice);
  
	cudaMalloc(&gpu_meshID, sizeof(int)  * numTranslations);
  cudaMalloc(&gpu_commandID, sizeof(int)  * numTranslations);
  cudaMalloc(&gpu_movementID, sizeof(int)  * numTranslations);
	cudaMalloc(&gpu_hOffset,sizeof(int) * (V+1));
	cudaMalloc(&gpu_hCsr,sizeof(int) * E);
	cudaMalloc(&gpu_hOpacity,sizeof(int) * V);
	cudaMalloc(&gpu_hGlobalCoordinatesX,sizeof(int) * V);
	cudaMalloc(&gpu_hGlobalCoordinatesY,sizeof(int) * V);
	cudaMalloc(&gpu_hFrameSizeX,sizeof(int)* V );
	cudaMalloc(&gpu_hFrameSizeY,sizeof(int)* V );
  cudaMalloc(&result,sizeof(int)*frameSizeX*frameSizeY);
  cudaMalloc(&gpu_postorder,sizeof(int)* V);
  cudaMalloc(&gpu_post_size,sizeof(int) * V);
  cudaMalloc(&gpu_post_offset,sizeof(int) * V);


	cudaMemcpy(gpu_hOffset,hOffset, sizeof(int) * (V+1) , cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_hCsr,hCsr, sizeof(int)*(E), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_hOpacity,hOpacity, sizeof(int)*(V), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_hGlobalCoordinatesX, hGlobalCoordinatesX ,sizeof(int)*(V), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_hGlobalCoordinatesY, hGlobalCoordinatesY ,sizeof(int)*(V), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_hFrameSizeX,hFrameSizeX, sizeof(int)*(V), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_hFrameSizeY,hFrameSizeY, sizeof(int)*(V), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_postorder,postorder, sizeof(int) * (V) , cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_post_size,postorder_size, sizeof(int) * (V) , cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_post_offset,offset, sizeof(int) * (V) , cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_meshID,meshID,sizeof(int)  * numTranslations, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_commandID,commandID,sizeof(int)  * numTranslations, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_movementID,movementId,sizeof(int)  * numTranslations, cudaMemcpyHostToDevice);
  

  
  free(dummy);
  free(offset);
  free(postorder);
  free(meshID);
  free(commandID);
  free(movementId);
  free(postorder_size);


  int blockdim=ceil(1.0*numTranslations/1024.0);
	kerneltranslation<<<blockdim,1024>>>(gpu_hOffset,gpu_hCsr,gpu_meshID,gpu_commandID,gpu_movementID,gpu_hGlobalCoordinatesX,gpu_hGlobalCoordinatesY,numTranslations,gpu_postorder,gpu_post_size,gpu_post_offset);
  cudaDeviceSynchronize();

  cudaMemset(result , -1 , sizeof(int) * frameSizeX * frameSizeY);
  int *maxi_opacity;
  cudaHostAlloc(&maxi_opacity,sizeof(int), 0);
  cudaMemset(maxi_opacity , -1 , sizeof(int) );

	blockdim=ceil((1.0*V)/1024);
  storing_Opacity<<<blockdim,1024>>>(result,gpu_hOpacity,gpu_hFrameSizeX,gpu_hFrameSizeY,gpu_hGlobalCoordinatesX,gpu_hGlobalCoordinatesY,frameSizeX,frameSizeY,V,maxi_opacity);
   cudaDeviceSynchronize();
   
  int value=*maxi_opacity;
   int *opacity_mapping;
   cudaMalloc(&opacity_mapping,sizeof(int)*(value+1));
   cudaMemset(opacity_mapping,-1,sizeof(int)*(value+1));
	blockdim=ceil((1.0*V)/1024);
  make_opacity_map<<<blockdim,1024 >>>(opacity_mapping,gpu_hOpacity,V);
  cudaDeviceSynchronize();

  blockdim=ceil((1.0*frameSizeX*frameSizeY)/1024);
	resultantScene<<<blockdim,1024>>>(result,gpu_hMesh,opacity_mapping,gpu_hGlobalCoordinatesX,gpu_hGlobalCoordinatesY,gpu_hFrameSizeX,gpu_hFrameSizeY,frameSizeX,frameSizeY);
  cudaMemcpy(hFinalPng,result,sizeof(int) * frameSizeX * frameSizeY,cudaMemcpyDeviceToHost);

  cudaFree(gpu_hOffset);
  cudaFree(gpu_hCsr);
  cudaFree(gpu_hOpacity);
  cudaFree(gpu_hGlobalCoordinatesX);
  cudaFree(gpu_hGlobalCoordinatesY);
  cudaFree(gpu_hFrameSizeX);
  cudaFree(gpu_hFrameSizeY);
  cudaFree(gpu_hMesh);
  cudaFree(gpu_meshID);
  cudaFree(gpu_commandID);
  cudaFree(gpu_movementID);
  cudaFree(result);


	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;

}
