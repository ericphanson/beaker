## Summary

- **Cutout Main Code**: 339 (Tests: 44)  
- **Head Main Code**: 477 (Tests: 0)  
- **Common Main Code**: 1012 (Tests: 1615)  
- **Total Main Code (excluding build.rs)**: 1828  
- **Total Main Code (including build.rs)**: 1978 (Reported: 1978)  
✅ Totals match.

## Raw Counts

File name: ./beaker/tests/integration_tests.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 0            | 0            | 0            | 0            | 0           
Tests        | 1253         | 226          | 4            | 97           | 1580        
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 1253         | 226          | 4            | 97           | 1580        

File name: ./beaker/build.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 150          | 37           | 0            | 15           | 202         
Tests        | 0            | 0            | 0            | 0            | 0           
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 150          | 37           | 0            | 15           | 202         

File name: ./beaker/src/yolo_preprocessing.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 51           | 10           | 0            | 9            | 70          
Tests        | 0            | 0            | 0            | 0            | 0           
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 51           | 10           | 0            | 9            | 70          

File name: ./beaker/src/head_detection.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 331          | 65           | 6            | 43           | 445         
Tests        | 0            | 0            | 0            | 0            | 0           
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 331          | 65           | 6            | 43           | 445         

File name: ./beaker/src/config.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 149          | 37           | 41           | 3            | 230         
Tests        | 101          | 15           | 0            | 4            | 120         
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 250          | 52           | 41           | 7            | 350         

File name: ./beaker/src/image_input.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 123          | 18           | 9            | 10           | 160         
Tests        | 97           | 21           | 0            | 9            | 127         
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 220          | 39           | 9            | 19           | 287         

File name: ./beaker/src/lib.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 13           | 1            | 4            | 0            | 18          
Tests        | 0            | 0            | 0            | 0            | 0           
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 13           | 1            | 4            | 0            | 18          

File name: ./beaker/src/output_manager.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 140          | 29           | 26           | 6            | 201         
Tests        | 112          | 26           | 0            | 0            | 138         
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 252          | 55           | 26           | 6            | 339         

File name: ./beaker/src/onnx_session.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 178          | 18           | 5            | 12           | 213         
Tests        | 0            | 0            | 0            | 0            | 0           
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 178          | 18           | 5            | 12           | 213         

File name: ./beaker/src/cutout_processing.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 159          | 33           | 4            | 15           | 211         
Tests        | 0            | 0            | 0            | 0            | 0           
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 159          | 33           | 4            | 15           | 211         

File name: ./beaker/src/model_cache.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 82           | 20           | 8            | 6            | 116         
Tests        | 25           | 4            | 0            | 0            | 29          
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 107          | 24           | 8            | 6            | 145         

File name: ./beaker/src/shared_metadata.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 47           | 9            | 4            | 1            | 61          
Tests        | 27           | 5            | 0            | 2            | 34          
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 74           | 14           | 4            | 3            | 95          

File name: ./beaker/src/main.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 188          | 19           | 3            | 10           | 220         
Tests        | 0            | 0            | 0            | 0            | 0           
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 188          | 19           | 3            | 10           | 220         

File name: ./beaker/src/yolo_postprocessing.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 95           | 22           | 0            | 8            | 125         
Tests        | 0            | 0            | 0            | 0            | 0           
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 95           | 22           | 0            | 8            | 125         

File name: ./beaker/src/model_processing.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 92           | 24           | 16           | 6            | 138         
Tests        | 0            | 0            | 0            | 0            | 0           
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 92           | 24           | 16           | 6            | 138         

File name: ./beaker/src/cutout_postprocessing.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 144          | 31           | 6            | 20           | 201         
Tests        | 30           | 6            | 0            | 4            | 40          
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 174          | 37           | 6            | 24           | 241         

File name: ./beaker/src/cutout_preprocessing.rs
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 36           | 10           | 1            | 7            | 54          
Tests        | 14           | 3            | 0            | 2            | 19          
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 50           | 13           | 1            | 9            | 73          

File count: 17
Type         | Code         | Blank        | Doc comments | Comments     | Total       
-------------|--------------|--------------|--------------|--------------|-------------
Main         | 1978         | 383          | 133          | 171          | 2665        
Tests        | 1659         | 306          | 4            | 118          | 2087        
Examples     | 0            | 0            | 0            | 0            | 0           
Total| 3637         | 689          | 137          | 289          | 4752        
