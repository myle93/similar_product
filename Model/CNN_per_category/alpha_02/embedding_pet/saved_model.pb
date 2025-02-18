׉ 
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8ת
�
conv2d_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_96/kernel
}
$conv2d_96/kernel/Read/ReadVariableOpReadVariableOpconv2d_96/kernel*&
_output_shapes
: *
dtype0
t
conv2d_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_96/bias
m
"conv2d_96/bias/Read/ReadVariableOpReadVariableOpconv2d_96/bias*
_output_shapes
: *
dtype0
�
batch_normalization_112/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_112/gamma
�
1batch_normalization_112/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_112/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_112/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_112/beta
�
0batch_normalization_112/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_112/beta*
_output_shapes
: *
dtype0
�
#batch_normalization_112/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_112/moving_mean
�
7batch_normalization_112/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_112/moving_mean*
_output_shapes
: *
dtype0
�
'batch_normalization_112/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_112/moving_variance
�
;batch_normalization_112/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_112/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_97/kernel
}
$conv2d_97/kernel/Read/ReadVariableOpReadVariableOpconv2d_97/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_97/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_97/bias
m
"conv2d_97/bias/Read/ReadVariableOpReadVariableOpconv2d_97/bias*
_output_shapes
: *
dtype0
�
batch_normalization_113/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_113/gamma
�
1batch_normalization_113/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_113/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_113/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_113/beta
�
0batch_normalization_113/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_113/beta*
_output_shapes
: *
dtype0
�
#batch_normalization_113/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_113/moving_mean
�
7batch_normalization_113/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_113/moving_mean*
_output_shapes
: *
dtype0
�
'batch_normalization_113/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_113/moving_variance
�
;batch_normalization_113/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_113/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_98/kernel
}
$conv2d_98/kernel/Read/ReadVariableOpReadVariableOpconv2d_98/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_98/bias
m
"conv2d_98/bias/Read/ReadVariableOpReadVariableOpconv2d_98/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_114/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_114/gamma
�
1batch_normalization_114/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_114/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_114/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_114/beta
�
0batch_normalization_114/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_114/beta*
_output_shapes
:@*
dtype0
�
#batch_normalization_114/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_114/moving_mean
�
7batch_normalization_114/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_114/moving_mean*
_output_shapes
:@*
dtype0
�
'batch_normalization_114/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_114/moving_variance
�
;batch_normalization_114/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_114/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_99/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_99/kernel
}
$conv2d_99/kernel/Read/ReadVariableOpReadVariableOpconv2d_99/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_99/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_99/bias
m
"conv2d_99/bias/Read/ReadVariableOpReadVariableOpconv2d_99/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_115/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_115/gamma
�
1batch_normalization_115/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_115/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_115/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_115/beta
�
0batch_normalization_115/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_115/beta*
_output_shapes
:@*
dtype0
�
#batch_normalization_115/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_115/moving_mean
�
7batch_normalization_115/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_115/moving_mean*
_output_shapes
:@*
dtype0
�
'batch_normalization_115/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_115/moving_variance
�
;batch_normalization_115/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_115/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_100/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*"
shared_nameconv2d_100/kernel
�
%conv2d_100/kernel/Read/ReadVariableOpReadVariableOpconv2d_100/kernel*'
_output_shapes
:@�*
dtype0
w
conv2d_100/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameconv2d_100/bias
p
#conv2d_100/bias/Read/ReadVariableOpReadVariableOpconv2d_100/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_116/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namebatch_normalization_116/gamma
�
1batch_normalization_116/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_116/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_116/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_116/beta
�
0batch_normalization_116/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_116/beta*
_output_shapes	
:�*
dtype0
�
#batch_normalization_116/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_116/moving_mean
�
7batch_normalization_116/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_116/moving_mean*
_output_shapes	
:�*
dtype0
�
'batch_normalization_116/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'batch_normalization_116/moving_variance
�
;batch_normalization_116/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_116/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_101/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*"
shared_nameconv2d_101/kernel
�
%conv2d_101/kernel/Read/ReadVariableOpReadVariableOpconv2d_101/kernel*(
_output_shapes
:��*
dtype0
w
conv2d_101/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameconv2d_101/bias
p
#conv2d_101/bias/Read/ReadVariableOpReadVariableOpconv2d_101/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_117/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namebatch_normalization_117/gamma
�
1batch_normalization_117/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_117/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_117/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_117/beta
�
0batch_normalization_117/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_117/beta*
_output_shapes	
:�*
dtype0
�
#batch_normalization_117/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_117/moving_mean
�
7batch_normalization_117/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_117/moving_mean*
_output_shapes	
:�*
dtype0
�
'batch_normalization_117/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'batch_normalization_117/moving_variance
�
;batch_normalization_117/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_117/moving_variance*
_output_shapes	
:�*
dtype0
�
batch_normalization_118/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namebatch_normalization_118/gamma
�
1batch_normalization_118/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_118/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_118/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_118/beta
�
0batch_normalization_118/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_118/beta*
_output_shapes	
:�*
dtype0
�
#batch_normalization_118/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_118/moving_mean
�
7batch_normalization_118/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_118/moving_mean*
_output_shapes	
:�*
dtype0
�
'batch_normalization_118/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'batch_normalization_118/moving_variance
�
;batch_normalization_118/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_118/moving_variance*
_output_shapes	
:�*
dtype0
|
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_17/kernel
u
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel* 
_output_shapes
:
��*
dtype0
s
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_17/bias
l
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes	
:�*
dtype0

NoOpNoOp
�d
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�d
value�dB�d B�d
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer-14
layer_with_weights-12
layer-15
layer-16
layer_with_weights-13
layer-17
layer-18
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
�
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
�
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3regularization_losses
4trainable_variables
5	variables
6	keras_api
R
7regularization_losses
8trainable_variables
9	variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
�
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
�
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
R
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
h

]kernel
^bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
�
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance
hregularization_losses
itrainable_variables
j	variables
k	keras_api
h

lkernel
mbias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
�
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
R
{regularization_losses
|trainable_variables
}	variables
~	keras_api
�
axis

�gamma
	�beta
�moving_mean
�moving_variance
�regularization_losses
�trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�trainable_variables
�	variables
�	keras_api
V
�regularization_losses
�trainable_variables
�	variables
�	keras_api
 
�
0
1
 2
!3
(4
)5
/6
07
;8
<9
B10
C11
J12
K13
Q14
R15
]16
^17
d18
e19
l20
m21
s22
t23
�24
�25
�26
�27
�
0
1
 2
!3
"4
#5
(6
)7
/8
09
110
211
;12
<13
B14
C15
D16
E17
J18
K19
Q20
R21
S22
T23
]24
^25
d26
e27
f28
g29
l30
m31
s32
t33
u34
v35
�36
�37
�38
�39
�40
�41
�
�layer_metrics
regularization_losses
trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
	variables
�layers
 
\Z
VARIABLE_VALUEconv2d_96/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_96/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
�layer_metrics
regularization_losses
trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
	variables
�layers
 
hf
VARIABLE_VALUEbatch_normalization_112/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_112/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_112/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_112/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
"2
#3
�
�layer_metrics
$regularization_losses
%trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
&	variables
�layers
\Z
VARIABLE_VALUEconv2d_97/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_97/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
�
�layer_metrics
*regularization_losses
+trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
,	variables
�layers
 
hf
VARIABLE_VALUEbatch_normalization_113/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_113/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_113/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_113/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
12
23
�
�layer_metrics
3regularization_losses
4trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
5	variables
�layers
 
 
 
�
�layer_metrics
7regularization_losses
8trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
9	variables
�layers
\Z
VARIABLE_VALUEconv2d_98/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_98/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
�
�layer_metrics
=regularization_losses
>trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
?	variables
�layers
 
hf
VARIABLE_VALUEbatch_normalization_114/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_114/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_114/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_114/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

B0
C1
D2
E3
�
�layer_metrics
Fregularization_losses
Gtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
H	variables
�layers
\Z
VARIABLE_VALUEconv2d_99/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_99/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
�
�layer_metrics
Lregularization_losses
Mtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
N	variables
�layers
 
hf
VARIABLE_VALUEbatch_normalization_115/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_115/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_115/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_115/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
S2
T3
�
�layer_metrics
Uregularization_losses
Vtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
W	variables
�layers
 
 
 
�
�layer_metrics
Yregularization_losses
Ztrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
[	variables
�layers
][
VARIABLE_VALUEconv2d_100/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_100/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
�
�layer_metrics
_regularization_losses
`trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
a	variables
�layers
 
hf
VARIABLE_VALUEbatch_normalization_116/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_116/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_116/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_116/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1

d0
e1
f2
g3
�
�layer_metrics
hregularization_losses
itrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
j	variables
�layers
^\
VARIABLE_VALUEconv2d_101/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_101/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

l0
m1
�
�layer_metrics
nregularization_losses
otrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
p	variables
�layers
 
ig
VARIABLE_VALUEbatch_normalization_117/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_117/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#batch_normalization_117/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'batch_normalization_117/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

s0
t1

s0
t1
u2
v3
�
�layer_metrics
wregularization_losses
xtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
y	variables
�layers
 
 
 
�
�layer_metrics
{regularization_losses
|trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
}	variables
�layers
 
ig
VARIABLE_VALUEbatch_normalization_118/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_118/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#batch_normalization_118/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'batch_normalization_118/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1
 
�0
�1
�2
�3
�
�layer_metrics
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
�	variables
�layers
 
 
 
�
�layer_metrics
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
�	variables
�layers
\Z
VARIABLE_VALUEdense_17/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_17/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�layer_metrics
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
�	variables
�layers
 
 
 
�
�layer_metrics
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
�	variables
�layers
 
 
h
"0
#1
12
23
D4
E5
S6
T7
f8
g9
u10
v11
�12
�13
 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
 
 
 
 
 
 
 

"0
#1
 
 
 
 
 
 
 
 
 

10
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 

D0
E1
 
 
 
 
 
 
 
 
 

S0
T1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

f0
g1
 
 
 
 
 
 
 
 
 

u0
v1
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_conv2d_96_inputPlaceholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_96_inputconv2d_96/kernelconv2d_96/biasbatch_normalization_112/gammabatch_normalization_112/beta#batch_normalization_112/moving_mean'batch_normalization_112/moving_varianceconv2d_97/kernelconv2d_97/biasbatch_normalization_113/gammabatch_normalization_113/beta#batch_normalization_113/moving_mean'batch_normalization_113/moving_varianceconv2d_98/kernelconv2d_98/biasbatch_normalization_114/gammabatch_normalization_114/beta#batch_normalization_114/moving_mean'batch_normalization_114/moving_varianceconv2d_99/kernelconv2d_99/biasbatch_normalization_115/gammabatch_normalization_115/beta#batch_normalization_115/moving_mean'batch_normalization_115/moving_varianceconv2d_100/kernelconv2d_100/biasbatch_normalization_116/gammabatch_normalization_116/beta#batch_normalization_116/moving_mean'batch_normalization_116/moving_varianceconv2d_101/kernelconv2d_101/biasbatch_normalization_117/gammabatch_normalization_117/beta#batch_normalization_117/moving_mean'batch_normalization_117/moving_variancebatch_normalization_118/gammabatch_normalization_118/beta#batch_normalization_118/moving_mean'batch_normalization_118/moving_variancedense_17/kerneldense_17/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_228665
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_96/kernel/Read/ReadVariableOp"conv2d_96/bias/Read/ReadVariableOp1batch_normalization_112/gamma/Read/ReadVariableOp0batch_normalization_112/beta/Read/ReadVariableOp7batch_normalization_112/moving_mean/Read/ReadVariableOp;batch_normalization_112/moving_variance/Read/ReadVariableOp$conv2d_97/kernel/Read/ReadVariableOp"conv2d_97/bias/Read/ReadVariableOp1batch_normalization_113/gamma/Read/ReadVariableOp0batch_normalization_113/beta/Read/ReadVariableOp7batch_normalization_113/moving_mean/Read/ReadVariableOp;batch_normalization_113/moving_variance/Read/ReadVariableOp$conv2d_98/kernel/Read/ReadVariableOp"conv2d_98/bias/Read/ReadVariableOp1batch_normalization_114/gamma/Read/ReadVariableOp0batch_normalization_114/beta/Read/ReadVariableOp7batch_normalization_114/moving_mean/Read/ReadVariableOp;batch_normalization_114/moving_variance/Read/ReadVariableOp$conv2d_99/kernel/Read/ReadVariableOp"conv2d_99/bias/Read/ReadVariableOp1batch_normalization_115/gamma/Read/ReadVariableOp0batch_normalization_115/beta/Read/ReadVariableOp7batch_normalization_115/moving_mean/Read/ReadVariableOp;batch_normalization_115/moving_variance/Read/ReadVariableOp%conv2d_100/kernel/Read/ReadVariableOp#conv2d_100/bias/Read/ReadVariableOp1batch_normalization_116/gamma/Read/ReadVariableOp0batch_normalization_116/beta/Read/ReadVariableOp7batch_normalization_116/moving_mean/Read/ReadVariableOp;batch_normalization_116/moving_variance/Read/ReadVariableOp%conv2d_101/kernel/Read/ReadVariableOp#conv2d_101/bias/Read/ReadVariableOp1batch_normalization_117/gamma/Read/ReadVariableOp0batch_normalization_117/beta/Read/ReadVariableOp7batch_normalization_117/moving_mean/Read/ReadVariableOp;batch_normalization_117/moving_variance/Read/ReadVariableOp1batch_normalization_118/gamma/Read/ReadVariableOp0batch_normalization_118/beta/Read/ReadVariableOp7batch_normalization_118/moving_mean/Read/ReadVariableOp;batch_normalization_118/moving_variance/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpConst*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_230411
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_96/kernelconv2d_96/biasbatch_normalization_112/gammabatch_normalization_112/beta#batch_normalization_112/moving_mean'batch_normalization_112/moving_varianceconv2d_97/kernelconv2d_97/biasbatch_normalization_113/gammabatch_normalization_113/beta#batch_normalization_113/moving_mean'batch_normalization_113/moving_varianceconv2d_98/kernelconv2d_98/biasbatch_normalization_114/gammabatch_normalization_114/beta#batch_normalization_114/moving_mean'batch_normalization_114/moving_varianceconv2d_99/kernelconv2d_99/biasbatch_normalization_115/gammabatch_normalization_115/beta#batch_normalization_115/moving_mean'batch_normalization_115/moving_varianceconv2d_100/kernelconv2d_100/biasbatch_normalization_116/gammabatch_normalization_116/beta#batch_normalization_116/moving_mean'batch_normalization_116/moving_varianceconv2d_101/kernelconv2d_101/biasbatch_normalization_117/gammabatch_normalization_117/beta#batch_normalization_117/moving_mean'batch_normalization_117/moving_variancebatch_normalization_118/gammabatch_normalization_118/beta#batch_normalization_118/moving_mean'batch_normalization_118/moving_variancedense_17/kerneldense_17/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_230547��
�

�
E__inference_conv2d_99_layer_call_and_return_conditional_losses_227633

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229223

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�k
�
I__inference_sequential_16_layer_call_and_return_conditional_losses_228074
conv2d_96_input
conv2d_96_227343
conv2d_96_227345"
batch_normalization_112_227412"
batch_normalization_112_227414"
batch_normalization_112_227416"
batch_normalization_112_227418
conv2d_97_227443
conv2d_97_227445"
batch_normalization_113_227512"
batch_normalization_113_227514"
batch_normalization_113_227516"
batch_normalization_113_227518
conv2d_98_227544
conv2d_98_227546"
batch_normalization_114_227613"
batch_normalization_114_227615"
batch_normalization_114_227617"
batch_normalization_114_227619
conv2d_99_227644
conv2d_99_227646"
batch_normalization_115_227713"
batch_normalization_115_227715"
batch_normalization_115_227717"
batch_normalization_115_227719
conv2d_100_227745
conv2d_100_227747"
batch_normalization_116_227814"
batch_normalization_116_227816"
batch_normalization_116_227818"
batch_normalization_116_227820
conv2d_101_227845
conv2d_101_227847"
batch_normalization_117_227914"
batch_normalization_117_227916"
batch_normalization_117_227918"
batch_normalization_117_227920"
batch_normalization_118_227988"
batch_normalization_118_227990"
batch_normalization_118_227992"
batch_normalization_118_227994
dense_17_228033
dense_17_228035
identity��/batch_normalization_112/StatefulPartitionedCall�/batch_normalization_113/StatefulPartitionedCall�/batch_normalization_114/StatefulPartitionedCall�/batch_normalization_115/StatefulPartitionedCall�/batch_normalization_116/StatefulPartitionedCall�/batch_normalization_117/StatefulPartitionedCall�/batch_normalization_118/StatefulPartitionedCall�"conv2d_100/StatefulPartitionedCall�"conv2d_101/StatefulPartitionedCall�!conv2d_96/StatefulPartitionedCall�!conv2d_97/StatefulPartitionedCall�!conv2d_98/StatefulPartitionedCall�!conv2d_99/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCallconv2d_96_inputconv2d_96_227343conv2d_96_227345*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_2273322#
!conv2d_96/StatefulPartitionedCall�
/batch_normalization_112/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0batch_normalization_112_227412batch_normalization_112_227414batch_normalization_112_227416batch_normalization_112_227418*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_22736721
/batch_normalization_112/StatefulPartitionedCall�
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_112/StatefulPartitionedCall:output:0conv2d_97_227443conv2d_97_227445*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_2274322#
!conv2d_97/StatefulPartitionedCall�
/batch_normalization_113/StatefulPartitionedCallStatefulPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0batch_normalization_113_227512batch_normalization_113_227514batch_normalization_113_227516batch_normalization_113_227518*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_22746721
/batch_normalization_113/StatefulPartitionedCall�
 max_pooling2d_48/PartitionedCallPartitionedCall8batch_normalization_113/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2267672"
 max_pooling2d_48/PartitionedCall�
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0conv2d_98_227544conv2d_98_227546*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_98_layer_call_and_return_conditional_losses_2275332#
!conv2d_98/StatefulPartitionedCall�
/batch_normalization_114/StatefulPartitionedCallStatefulPartitionedCall*conv2d_98/StatefulPartitionedCall:output:0batch_normalization_114_227613batch_normalization_114_227615batch_normalization_114_227617batch_normalization_114_227619*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_22756821
/batch_normalization_114/StatefulPartitionedCall�
!conv2d_99/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_114/StatefulPartitionedCall:output:0conv2d_99_227644conv2d_99_227646*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_99_layer_call_and_return_conditional_losses_2276332#
!conv2d_99/StatefulPartitionedCall�
/batch_normalization_115/StatefulPartitionedCallStatefulPartitionedCall*conv2d_99/StatefulPartitionedCall:output:0batch_normalization_115_227713batch_normalization_115_227715batch_normalization_115_227717batch_normalization_115_227719*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_22766821
/batch_normalization_115/StatefulPartitionedCall�
 max_pooling2d_49/PartitionedCallPartitionedCall8batch_normalization_115/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2269872"
 max_pooling2d_49/PartitionedCall�
"conv2d_100/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_100_227745conv2d_100_227747*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_100_layer_call_and_return_conditional_losses_2277342$
"conv2d_100/StatefulPartitionedCall�
/batch_normalization_116/StatefulPartitionedCallStatefulPartitionedCall+conv2d_100/StatefulPartitionedCall:output:0batch_normalization_116_227814batch_normalization_116_227816batch_normalization_116_227818batch_normalization_116_227820*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_22776921
/batch_normalization_116/StatefulPartitionedCall�
"conv2d_101/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_116/StatefulPartitionedCall:output:0conv2d_101_227845conv2d_101_227847*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_101_layer_call_and_return_conditional_losses_2278342$
"conv2d_101/StatefulPartitionedCall�
/batch_normalization_117/StatefulPartitionedCallStatefulPartitionedCall+conv2d_101/StatefulPartitionedCall:output:0batch_normalization_117_227914batch_normalization_117_227916batch_normalization_117_227918batch_normalization_117_227920*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_22786921
/batch_normalization_117/StatefulPartitionedCall�
 max_pooling2d_50/PartitionedCallPartitionedCall8batch_normalization_117/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2272072"
 max_pooling2d_50/PartitionedCall�
/batch_normalization_118/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0batch_normalization_118_227988batch_normalization_118_227990batch_normalization_118_227992batch_normalization_118_227994*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_22794321
/batch_normalization_118/StatefulPartitionedCall�
flatten_16/PartitionedCallPartitionedCall8batch_normalization_118/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_2280032
flatten_16/PartitionedCall�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_17_228033dense_17_228035*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_2280222"
 dense_17/StatefulPartitionedCall�
lambda_17/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lambda_17_layer_call_and_return_conditional_losses_2280492
lambda_17/PartitionedCall�
IdentityIdentity"lambda_17/PartitionedCall:output:00^batch_normalization_112/StatefulPartitionedCall0^batch_normalization_113/StatefulPartitionedCall0^batch_normalization_114/StatefulPartitionedCall0^batch_normalization_115/StatefulPartitionedCall0^batch_normalization_116/StatefulPartitionedCall0^batch_normalization_117/StatefulPartitionedCall0^batch_normalization_118/StatefulPartitionedCall#^conv2d_100/StatefulPartitionedCall#^conv2d_101/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall"^conv2d_98/StatefulPartitionedCall"^conv2d_99/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2b
/batch_normalization_112/StatefulPartitionedCall/batch_normalization_112/StatefulPartitionedCall2b
/batch_normalization_113/StatefulPartitionedCall/batch_normalization_113/StatefulPartitionedCall2b
/batch_normalization_114/StatefulPartitionedCall/batch_normalization_114/StatefulPartitionedCall2b
/batch_normalization_115/StatefulPartitionedCall/batch_normalization_115/StatefulPartitionedCall2b
/batch_normalization_116/StatefulPartitionedCall/batch_normalization_116/StatefulPartitionedCall2b
/batch_normalization_117/StatefulPartitionedCall/batch_normalization_117/StatefulPartitionedCall2b
/batch_normalization_118/StatefulPartitionedCall/batch_normalization_118/StatefulPartitionedCall2H
"conv2d_100/StatefulPartitionedCall"conv2d_100/StatefulPartitionedCall2H
"conv2d_101/StatefulPartitionedCall"conv2d_101/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall2F
!conv2d_99/StatefulPartitionedCall!conv2d_99/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_96_input
�
�
8__inference_batch_normalization_116_layer_call_fn_229910

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_2277692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_226615

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_116_layer_call_fn_229923

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_2277872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229601

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�+
!__inference__wrapped_model_226553
conv2d_96_input:
6sequential_16_conv2d_96_conv2d_readvariableop_resource;
7sequential_16_conv2d_96_biasadd_readvariableop_resourceA
=sequential_16_batch_normalization_112_readvariableop_resourceC
?sequential_16_batch_normalization_112_readvariableop_1_resourceR
Nsequential_16_batch_normalization_112_fusedbatchnormv3_readvariableop_resourceT
Psequential_16_batch_normalization_112_fusedbatchnormv3_readvariableop_1_resource:
6sequential_16_conv2d_97_conv2d_readvariableop_resource;
7sequential_16_conv2d_97_biasadd_readvariableop_resourceA
=sequential_16_batch_normalization_113_readvariableop_resourceC
?sequential_16_batch_normalization_113_readvariableop_1_resourceR
Nsequential_16_batch_normalization_113_fusedbatchnormv3_readvariableop_resourceT
Psequential_16_batch_normalization_113_fusedbatchnormv3_readvariableop_1_resource:
6sequential_16_conv2d_98_conv2d_readvariableop_resource;
7sequential_16_conv2d_98_biasadd_readvariableop_resourceA
=sequential_16_batch_normalization_114_readvariableop_resourceC
?sequential_16_batch_normalization_114_readvariableop_1_resourceR
Nsequential_16_batch_normalization_114_fusedbatchnormv3_readvariableop_resourceT
Psequential_16_batch_normalization_114_fusedbatchnormv3_readvariableop_1_resource:
6sequential_16_conv2d_99_conv2d_readvariableop_resource;
7sequential_16_conv2d_99_biasadd_readvariableop_resourceA
=sequential_16_batch_normalization_115_readvariableop_resourceC
?sequential_16_batch_normalization_115_readvariableop_1_resourceR
Nsequential_16_batch_normalization_115_fusedbatchnormv3_readvariableop_resourceT
Psequential_16_batch_normalization_115_fusedbatchnormv3_readvariableop_1_resource;
7sequential_16_conv2d_100_conv2d_readvariableop_resource<
8sequential_16_conv2d_100_biasadd_readvariableop_resourceA
=sequential_16_batch_normalization_116_readvariableop_resourceC
?sequential_16_batch_normalization_116_readvariableop_1_resourceR
Nsequential_16_batch_normalization_116_fusedbatchnormv3_readvariableop_resourceT
Psequential_16_batch_normalization_116_fusedbatchnormv3_readvariableop_1_resource;
7sequential_16_conv2d_101_conv2d_readvariableop_resource<
8sequential_16_conv2d_101_biasadd_readvariableop_resourceA
=sequential_16_batch_normalization_117_readvariableop_resourceC
?sequential_16_batch_normalization_117_readvariableop_1_resourceR
Nsequential_16_batch_normalization_117_fusedbatchnormv3_readvariableop_resourceT
Psequential_16_batch_normalization_117_fusedbatchnormv3_readvariableop_1_resourceA
=sequential_16_batch_normalization_118_readvariableop_resourceC
?sequential_16_batch_normalization_118_readvariableop_1_resourceR
Nsequential_16_batch_normalization_118_fusedbatchnormv3_readvariableop_resourceT
Psequential_16_batch_normalization_118_fusedbatchnormv3_readvariableop_1_resource9
5sequential_16_dense_17_matmul_readvariableop_resource:
6sequential_16_dense_17_biasadd_readvariableop_resource
identity��Esequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp�Gsequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1�4sequential_16/batch_normalization_112/ReadVariableOp�6sequential_16/batch_normalization_112/ReadVariableOp_1�Esequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp�Gsequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1�4sequential_16/batch_normalization_113/ReadVariableOp�6sequential_16/batch_normalization_113/ReadVariableOp_1�Esequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp�Gsequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1�4sequential_16/batch_normalization_114/ReadVariableOp�6sequential_16/batch_normalization_114/ReadVariableOp_1�Esequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp�Gsequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1�4sequential_16/batch_normalization_115/ReadVariableOp�6sequential_16/batch_normalization_115/ReadVariableOp_1�Esequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp�Gsequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1�4sequential_16/batch_normalization_116/ReadVariableOp�6sequential_16/batch_normalization_116/ReadVariableOp_1�Esequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp�Gsequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1�4sequential_16/batch_normalization_117/ReadVariableOp�6sequential_16/batch_normalization_117/ReadVariableOp_1�Esequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp�Gsequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1�4sequential_16/batch_normalization_118/ReadVariableOp�6sequential_16/batch_normalization_118/ReadVariableOp_1�/sequential_16/conv2d_100/BiasAdd/ReadVariableOp�.sequential_16/conv2d_100/Conv2D/ReadVariableOp�/sequential_16/conv2d_101/BiasAdd/ReadVariableOp�.sequential_16/conv2d_101/Conv2D/ReadVariableOp�.sequential_16/conv2d_96/BiasAdd/ReadVariableOp�-sequential_16/conv2d_96/Conv2D/ReadVariableOp�.sequential_16/conv2d_97/BiasAdd/ReadVariableOp�-sequential_16/conv2d_97/Conv2D/ReadVariableOp�.sequential_16/conv2d_98/BiasAdd/ReadVariableOp�-sequential_16/conv2d_98/Conv2D/ReadVariableOp�.sequential_16/conv2d_99/BiasAdd/ReadVariableOp�-sequential_16/conv2d_99/Conv2D/ReadVariableOp�-sequential_16/dense_17/BiasAdd/ReadVariableOp�,sequential_16/dense_17/MatMul/ReadVariableOp�
-sequential_16/conv2d_96/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_16/conv2d_96/Conv2D/ReadVariableOp�
sequential_16/conv2d_96/Conv2DConv2Dconv2d_96_input5sequential_16/conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2 
sequential_16/conv2d_96/Conv2D�
.sequential_16/conv2d_96/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_96/BiasAdd/ReadVariableOp�
sequential_16/conv2d_96/BiasAddBiasAdd'sequential_16/conv2d_96/Conv2D:output:06sequential_16/conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2!
sequential_16/conv2d_96/BiasAdd�
sequential_16/conv2d_96/ReluRelu(sequential_16/conv2d_96/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
sequential_16/conv2d_96/Relu�
4sequential_16/batch_normalization_112/ReadVariableOpReadVariableOp=sequential_16_batch_normalization_112_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_16/batch_normalization_112/ReadVariableOp�
6sequential_16/batch_normalization_112/ReadVariableOp_1ReadVariableOp?sequential_16_batch_normalization_112_readvariableop_1_resource*
_output_shapes
: *
dtype028
6sequential_16/batch_normalization_112/ReadVariableOp_1�
Esequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_16_batch_normalization_112_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02G
Esequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp�
Gsequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_16_batch_normalization_112_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gsequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1�
6sequential_16/batch_normalization_112/FusedBatchNormV3FusedBatchNormV3*sequential_16/conv2d_96/Relu:activations:0<sequential_16/batch_normalization_112/ReadVariableOp:value:0>sequential_16/batch_normalization_112/ReadVariableOp_1:value:0Msequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp:value:0Osequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 28
6sequential_16/batch_normalization_112/FusedBatchNormV3�
-sequential_16/conv2d_97/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_97_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02/
-sequential_16/conv2d_97/Conv2D/ReadVariableOp�
sequential_16/conv2d_97/Conv2DConv2D:sequential_16/batch_normalization_112/FusedBatchNormV3:y:05sequential_16/conv2d_97/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2 
sequential_16/conv2d_97/Conv2D�
.sequential_16/conv2d_97/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_97_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_97/BiasAdd/ReadVariableOp�
sequential_16/conv2d_97/BiasAddBiasAdd'sequential_16/conv2d_97/Conv2D:output:06sequential_16/conv2d_97/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2!
sequential_16/conv2d_97/BiasAdd�
sequential_16/conv2d_97/ReluRelu(sequential_16/conv2d_97/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
sequential_16/conv2d_97/Relu�
4sequential_16/batch_normalization_113/ReadVariableOpReadVariableOp=sequential_16_batch_normalization_113_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_16/batch_normalization_113/ReadVariableOp�
6sequential_16/batch_normalization_113/ReadVariableOp_1ReadVariableOp?sequential_16_batch_normalization_113_readvariableop_1_resource*
_output_shapes
: *
dtype028
6sequential_16/batch_normalization_113/ReadVariableOp_1�
Esequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_16_batch_normalization_113_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02G
Esequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp�
Gsequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_16_batch_normalization_113_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gsequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1�
6sequential_16/batch_normalization_113/FusedBatchNormV3FusedBatchNormV3*sequential_16/conv2d_97/Relu:activations:0<sequential_16/batch_normalization_113/ReadVariableOp:value:0>sequential_16/batch_normalization_113/ReadVariableOp_1:value:0Msequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp:value:0Osequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 28
6sequential_16/batch_normalization_113/FusedBatchNormV3�
&sequential_16/max_pooling2d_48/MaxPoolMaxPool:sequential_16/batch_normalization_113/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_48/MaxPool�
-sequential_16/conv2d_98/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_98_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_16/conv2d_98/Conv2D/ReadVariableOp�
sequential_16/conv2d_98/Conv2DConv2D/sequential_16/max_pooling2d_48/MaxPool:output:05sequential_16/conv2d_98/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2 
sequential_16/conv2d_98/Conv2D�
.sequential_16/conv2d_98/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_16/conv2d_98/BiasAdd/ReadVariableOp�
sequential_16/conv2d_98/BiasAddBiasAdd'sequential_16/conv2d_98/Conv2D:output:06sequential_16/conv2d_98/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2!
sequential_16/conv2d_98/BiasAdd�
sequential_16/conv2d_98/ReluRelu(sequential_16/conv2d_98/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential_16/conv2d_98/Relu�
4sequential_16/batch_normalization_114/ReadVariableOpReadVariableOp=sequential_16_batch_normalization_114_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential_16/batch_normalization_114/ReadVariableOp�
6sequential_16/batch_normalization_114/ReadVariableOp_1ReadVariableOp?sequential_16_batch_normalization_114_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6sequential_16/batch_normalization_114/ReadVariableOp_1�
Esequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_16_batch_normalization_114_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02G
Esequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp�
Gsequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_16_batch_normalization_114_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02I
Gsequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1�
6sequential_16/batch_normalization_114/FusedBatchNormV3FusedBatchNormV3*sequential_16/conv2d_98/Relu:activations:0<sequential_16/batch_normalization_114/ReadVariableOp:value:0>sequential_16/batch_normalization_114/ReadVariableOp_1:value:0Msequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp:value:0Osequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 28
6sequential_16/batch_normalization_114/FusedBatchNormV3�
-sequential_16/conv2d_99/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_99_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_16/conv2d_99/Conv2D/ReadVariableOp�
sequential_16/conv2d_99/Conv2DConv2D:sequential_16/batch_normalization_114/FusedBatchNormV3:y:05sequential_16/conv2d_99/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2 
sequential_16/conv2d_99/Conv2D�
.sequential_16/conv2d_99/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_16/conv2d_99/BiasAdd/ReadVariableOp�
sequential_16/conv2d_99/BiasAddBiasAdd'sequential_16/conv2d_99/Conv2D:output:06sequential_16/conv2d_99/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2!
sequential_16/conv2d_99/BiasAdd�
sequential_16/conv2d_99/ReluRelu(sequential_16/conv2d_99/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential_16/conv2d_99/Relu�
4sequential_16/batch_normalization_115/ReadVariableOpReadVariableOp=sequential_16_batch_normalization_115_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential_16/batch_normalization_115/ReadVariableOp�
6sequential_16/batch_normalization_115/ReadVariableOp_1ReadVariableOp?sequential_16_batch_normalization_115_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6sequential_16/batch_normalization_115/ReadVariableOp_1�
Esequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_16_batch_normalization_115_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02G
Esequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp�
Gsequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_16_batch_normalization_115_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02I
Gsequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1�
6sequential_16/batch_normalization_115/FusedBatchNormV3FusedBatchNormV3*sequential_16/conv2d_99/Relu:activations:0<sequential_16/batch_normalization_115/ReadVariableOp:value:0>sequential_16/batch_normalization_115/ReadVariableOp_1:value:0Msequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp:value:0Osequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 28
6sequential_16/batch_normalization_115/FusedBatchNormV3�
&sequential_16/max_pooling2d_49/MaxPoolMaxPool:sequential_16/batch_normalization_115/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_49/MaxPool�
.sequential_16/conv2d_100/Conv2D/ReadVariableOpReadVariableOp7sequential_16_conv2d_100_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype020
.sequential_16/conv2d_100/Conv2D/ReadVariableOp�
sequential_16/conv2d_100/Conv2DConv2D/sequential_16/max_pooling2d_49/MaxPool:output:06sequential_16/conv2d_100/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2!
sequential_16/conv2d_100/Conv2D�
/sequential_16/conv2d_100/BiasAdd/ReadVariableOpReadVariableOp8sequential_16_conv2d_100_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype021
/sequential_16/conv2d_100/BiasAdd/ReadVariableOp�
 sequential_16/conv2d_100/BiasAddBiasAdd(sequential_16/conv2d_100/Conv2D:output:07sequential_16/conv2d_100/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2"
 sequential_16/conv2d_100/BiasAdd�
sequential_16/conv2d_100/ReluRelu)sequential_16/conv2d_100/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_16/conv2d_100/Relu�
4sequential_16/batch_normalization_116/ReadVariableOpReadVariableOp=sequential_16_batch_normalization_116_readvariableop_resource*
_output_shapes	
:�*
dtype026
4sequential_16/batch_normalization_116/ReadVariableOp�
6sequential_16/batch_normalization_116/ReadVariableOp_1ReadVariableOp?sequential_16_batch_normalization_116_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6sequential_16/batch_normalization_116/ReadVariableOp_1�
Esequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_16_batch_normalization_116_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02G
Esequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp�
Gsequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_16_batch_normalization_116_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02I
Gsequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1�
6sequential_16/batch_normalization_116/FusedBatchNormV3FusedBatchNormV3+sequential_16/conv2d_100/Relu:activations:0<sequential_16/batch_normalization_116/ReadVariableOp:value:0>sequential_16/batch_normalization_116/ReadVariableOp_1:value:0Msequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp:value:0Osequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 28
6sequential_16/batch_normalization_116/FusedBatchNormV3�
.sequential_16/conv2d_101/Conv2D/ReadVariableOpReadVariableOp7sequential_16_conv2d_101_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype020
.sequential_16/conv2d_101/Conv2D/ReadVariableOp�
sequential_16/conv2d_101/Conv2DConv2D:sequential_16/batch_normalization_116/FusedBatchNormV3:y:06sequential_16/conv2d_101/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2!
sequential_16/conv2d_101/Conv2D�
/sequential_16/conv2d_101/BiasAdd/ReadVariableOpReadVariableOp8sequential_16_conv2d_101_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype021
/sequential_16/conv2d_101/BiasAdd/ReadVariableOp�
 sequential_16/conv2d_101/BiasAddBiasAdd(sequential_16/conv2d_101/Conv2D:output:07sequential_16/conv2d_101/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2"
 sequential_16/conv2d_101/BiasAdd�
sequential_16/conv2d_101/ReluRelu)sequential_16/conv2d_101/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_16/conv2d_101/Relu�
4sequential_16/batch_normalization_117/ReadVariableOpReadVariableOp=sequential_16_batch_normalization_117_readvariableop_resource*
_output_shapes	
:�*
dtype026
4sequential_16/batch_normalization_117/ReadVariableOp�
6sequential_16/batch_normalization_117/ReadVariableOp_1ReadVariableOp?sequential_16_batch_normalization_117_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6sequential_16/batch_normalization_117/ReadVariableOp_1�
Esequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_16_batch_normalization_117_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02G
Esequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp�
Gsequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_16_batch_normalization_117_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02I
Gsequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1�
6sequential_16/batch_normalization_117/FusedBatchNormV3FusedBatchNormV3+sequential_16/conv2d_101/Relu:activations:0<sequential_16/batch_normalization_117/ReadVariableOp:value:0>sequential_16/batch_normalization_117/ReadVariableOp_1:value:0Msequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp:value:0Osequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 28
6sequential_16/batch_normalization_117/FusedBatchNormV3�
&sequential_16/max_pooling2d_50/MaxPoolMaxPool:sequential_16/batch_normalization_117/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_50/MaxPool�
4sequential_16/batch_normalization_118/ReadVariableOpReadVariableOp=sequential_16_batch_normalization_118_readvariableop_resource*
_output_shapes	
:�*
dtype026
4sequential_16/batch_normalization_118/ReadVariableOp�
6sequential_16/batch_normalization_118/ReadVariableOp_1ReadVariableOp?sequential_16_batch_normalization_118_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6sequential_16/batch_normalization_118/ReadVariableOp_1�
Esequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_16_batch_normalization_118_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02G
Esequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp�
Gsequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_16_batch_normalization_118_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02I
Gsequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1�
6sequential_16/batch_normalization_118/FusedBatchNormV3FusedBatchNormV3/sequential_16/max_pooling2d_50/MaxPool:output:0<sequential_16/batch_normalization_118/ReadVariableOp:value:0>sequential_16/batch_normalization_118/ReadVariableOp_1:value:0Msequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp:value:0Osequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 28
6sequential_16/batch_normalization_118/FusedBatchNormV3�
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2 
sequential_16/flatten_16/Const�
 sequential_16/flatten_16/ReshapeReshape:sequential_16/batch_normalization_118/FusedBatchNormV3:y:0'sequential_16/flatten_16/Const:output:0*
T0*(
_output_shapes
:����������2"
 sequential_16/flatten_16/Reshape�
,sequential_16/dense_17/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,sequential_16/dense_17/MatMul/ReadVariableOp�
sequential_16/dense_17/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_16/dense_17/MatMul�
-sequential_16/dense_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_16/dense_17/BiasAdd/ReadVariableOp�
sequential_16/dense_17/BiasAddBiasAdd'sequential_16/dense_17/MatMul:product:05sequential_16/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_16/dense_17/BiasAdd�
sequential_16/dense_17/SigmoidSigmoid'sequential_16/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:����������2 
sequential_16/dense_17/Sigmoid�
+sequential_16/lambda_17/l2_normalize/SquareSquare"sequential_16/dense_17/Sigmoid:y:0*
T0*(
_output_shapes
:����������2-
+sequential_16/lambda_17/l2_normalize/Square�
:sequential_16/lambda_17/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_16/lambda_17/l2_normalize/Sum/reduction_indices�
(sequential_16/lambda_17/l2_normalize/SumSum/sequential_16/lambda_17/l2_normalize/Square:y:0Csequential_16/lambda_17/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2*
(sequential_16/lambda_17/l2_normalize/Sum�
.sequential_16/lambda_17/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+20
.sequential_16/lambda_17/l2_normalize/Maximum/y�
,sequential_16/lambda_17/l2_normalize/MaximumMaximum1sequential_16/lambda_17/l2_normalize/Sum:output:07sequential_16/lambda_17/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2.
,sequential_16/lambda_17/l2_normalize/Maximum�
*sequential_16/lambda_17/l2_normalize/RsqrtRsqrt0sequential_16/lambda_17/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2,
*sequential_16/lambda_17/l2_normalize/Rsqrt�
$sequential_16/lambda_17/l2_normalizeMul"sequential_16/dense_17/Sigmoid:y:0.sequential_16/lambda_17/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2&
$sequential_16/lambda_17/l2_normalize�
IdentityIdentity(sequential_16/lambda_17/l2_normalize:z:0F^sequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOpH^sequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_15^sequential_16/batch_normalization_112/ReadVariableOp7^sequential_16/batch_normalization_112/ReadVariableOp_1F^sequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOpH^sequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp_15^sequential_16/batch_normalization_113/ReadVariableOp7^sequential_16/batch_normalization_113/ReadVariableOp_1F^sequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOpH^sequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp_15^sequential_16/batch_normalization_114/ReadVariableOp7^sequential_16/batch_normalization_114/ReadVariableOp_1F^sequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOpH^sequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp_15^sequential_16/batch_normalization_115/ReadVariableOp7^sequential_16/batch_normalization_115/ReadVariableOp_1F^sequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOpH^sequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp_15^sequential_16/batch_normalization_116/ReadVariableOp7^sequential_16/batch_normalization_116/ReadVariableOp_1F^sequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOpH^sequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp_15^sequential_16/batch_normalization_117/ReadVariableOp7^sequential_16/batch_normalization_117/ReadVariableOp_1F^sequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOpH^sequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp_15^sequential_16/batch_normalization_118/ReadVariableOp7^sequential_16/batch_normalization_118/ReadVariableOp_10^sequential_16/conv2d_100/BiasAdd/ReadVariableOp/^sequential_16/conv2d_100/Conv2D/ReadVariableOp0^sequential_16/conv2d_101/BiasAdd/ReadVariableOp/^sequential_16/conv2d_101/Conv2D/ReadVariableOp/^sequential_16/conv2d_96/BiasAdd/ReadVariableOp.^sequential_16/conv2d_96/Conv2D/ReadVariableOp/^sequential_16/conv2d_97/BiasAdd/ReadVariableOp.^sequential_16/conv2d_97/Conv2D/ReadVariableOp/^sequential_16/conv2d_98/BiasAdd/ReadVariableOp.^sequential_16/conv2d_98/Conv2D/ReadVariableOp/^sequential_16/conv2d_99/BiasAdd/ReadVariableOp.^sequential_16/conv2d_99/Conv2D/ReadVariableOp.^sequential_16/dense_17/BiasAdd/ReadVariableOp-^sequential_16/dense_17/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2�
Esequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOpEsequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp2�
Gsequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1Gsequential_16/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_12l
4sequential_16/batch_normalization_112/ReadVariableOp4sequential_16/batch_normalization_112/ReadVariableOp2p
6sequential_16/batch_normalization_112/ReadVariableOp_16sequential_16/batch_normalization_112/ReadVariableOp_12�
Esequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOpEsequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp2�
Gsequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1Gsequential_16/batch_normalization_113/FusedBatchNormV3/ReadVariableOp_12l
4sequential_16/batch_normalization_113/ReadVariableOp4sequential_16/batch_normalization_113/ReadVariableOp2p
6sequential_16/batch_normalization_113/ReadVariableOp_16sequential_16/batch_normalization_113/ReadVariableOp_12�
Esequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOpEsequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp2�
Gsequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1Gsequential_16/batch_normalization_114/FusedBatchNormV3/ReadVariableOp_12l
4sequential_16/batch_normalization_114/ReadVariableOp4sequential_16/batch_normalization_114/ReadVariableOp2p
6sequential_16/batch_normalization_114/ReadVariableOp_16sequential_16/batch_normalization_114/ReadVariableOp_12�
Esequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOpEsequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp2�
Gsequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1Gsequential_16/batch_normalization_115/FusedBatchNormV3/ReadVariableOp_12l
4sequential_16/batch_normalization_115/ReadVariableOp4sequential_16/batch_normalization_115/ReadVariableOp2p
6sequential_16/batch_normalization_115/ReadVariableOp_16sequential_16/batch_normalization_115/ReadVariableOp_12�
Esequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOpEsequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp2�
Gsequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1Gsequential_16/batch_normalization_116/FusedBatchNormV3/ReadVariableOp_12l
4sequential_16/batch_normalization_116/ReadVariableOp4sequential_16/batch_normalization_116/ReadVariableOp2p
6sequential_16/batch_normalization_116/ReadVariableOp_16sequential_16/batch_normalization_116/ReadVariableOp_12�
Esequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOpEsequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp2�
Gsequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1Gsequential_16/batch_normalization_117/FusedBatchNormV3/ReadVariableOp_12l
4sequential_16/batch_normalization_117/ReadVariableOp4sequential_16/batch_normalization_117/ReadVariableOp2p
6sequential_16/batch_normalization_117/ReadVariableOp_16sequential_16/batch_normalization_117/ReadVariableOp_12�
Esequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOpEsequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp2�
Gsequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1Gsequential_16/batch_normalization_118/FusedBatchNormV3/ReadVariableOp_12l
4sequential_16/batch_normalization_118/ReadVariableOp4sequential_16/batch_normalization_118/ReadVariableOp2p
6sequential_16/batch_normalization_118/ReadVariableOp_16sequential_16/batch_normalization_118/ReadVariableOp_12b
/sequential_16/conv2d_100/BiasAdd/ReadVariableOp/sequential_16/conv2d_100/BiasAdd/ReadVariableOp2`
.sequential_16/conv2d_100/Conv2D/ReadVariableOp.sequential_16/conv2d_100/Conv2D/ReadVariableOp2b
/sequential_16/conv2d_101/BiasAdd/ReadVariableOp/sequential_16/conv2d_101/BiasAdd/ReadVariableOp2`
.sequential_16/conv2d_101/Conv2D/ReadVariableOp.sequential_16/conv2d_101/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_96/BiasAdd/ReadVariableOp.sequential_16/conv2d_96/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_96/Conv2D/ReadVariableOp-sequential_16/conv2d_96/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_97/BiasAdd/ReadVariableOp.sequential_16/conv2d_97/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_97/Conv2D/ReadVariableOp-sequential_16/conv2d_97/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_98/BiasAdd/ReadVariableOp.sequential_16/conv2d_98/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_98/Conv2D/ReadVariableOp-sequential_16/conv2d_98/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_99/BiasAdd/ReadVariableOp.sequential_16/conv2d_99/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_99/Conv2D/ReadVariableOp-sequential_16/conv2d_99/Conv2D/ReadVariableOp2^
-sequential_16/dense_17/BiasAdd/ReadVariableOp-sequential_16/dense_17/BiasAdd/ReadVariableOp2\
,sequential_16/dense_17/MatMul/ReadVariableOp,sequential_16/dense_17/MatMul/ReadVariableOp:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_96_input
�	
�
D__inference_dense_17_layer_call_and_return_conditional_losses_228022

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_226835

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229305

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
��
�'
I__inference_sequential_16_layer_call_and_return_conditional_losses_228842

inputs,
(conv2d_96_conv2d_readvariableop_resource-
)conv2d_96_biasadd_readvariableop_resource3
/batch_normalization_112_readvariableop_resource5
1batch_normalization_112_readvariableop_1_resourceD
@batch_normalization_112_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_112_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_97_conv2d_readvariableop_resource-
)conv2d_97_biasadd_readvariableop_resource3
/batch_normalization_113_readvariableop_resource5
1batch_normalization_113_readvariableop_1_resourceD
@batch_normalization_113_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_113_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_98_conv2d_readvariableop_resource-
)conv2d_98_biasadd_readvariableop_resource3
/batch_normalization_114_readvariableop_resource5
1batch_normalization_114_readvariableop_1_resourceD
@batch_normalization_114_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_114_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_99_conv2d_readvariableop_resource-
)conv2d_99_biasadd_readvariableop_resource3
/batch_normalization_115_readvariableop_resource5
1batch_normalization_115_readvariableop_1_resourceD
@batch_normalization_115_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_115_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_100_conv2d_readvariableop_resource.
*conv2d_100_biasadd_readvariableop_resource3
/batch_normalization_116_readvariableop_resource5
1batch_normalization_116_readvariableop_1_resourceD
@batch_normalization_116_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_116_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_101_conv2d_readvariableop_resource.
*conv2d_101_biasadd_readvariableop_resource3
/batch_normalization_117_readvariableop_resource5
1batch_normalization_117_readvariableop_1_resourceD
@batch_normalization_117_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_117_fusedbatchnormv3_readvariableop_1_resource3
/batch_normalization_118_readvariableop_resource5
1batch_normalization_118_readvariableop_1_resourceD
@batch_normalization_118_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_118_fusedbatchnormv3_readvariableop_1_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity��&batch_normalization_112/AssignNewValue�(batch_normalization_112/AssignNewValue_1�7batch_normalization_112/FusedBatchNormV3/ReadVariableOp�9batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_112/ReadVariableOp�(batch_normalization_112/ReadVariableOp_1�&batch_normalization_113/AssignNewValue�(batch_normalization_113/AssignNewValue_1�7batch_normalization_113/FusedBatchNormV3/ReadVariableOp�9batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_113/ReadVariableOp�(batch_normalization_113/ReadVariableOp_1�&batch_normalization_114/AssignNewValue�(batch_normalization_114/AssignNewValue_1�7batch_normalization_114/FusedBatchNormV3/ReadVariableOp�9batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_114/ReadVariableOp�(batch_normalization_114/ReadVariableOp_1�&batch_normalization_115/AssignNewValue�(batch_normalization_115/AssignNewValue_1�7batch_normalization_115/FusedBatchNormV3/ReadVariableOp�9batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_115/ReadVariableOp�(batch_normalization_115/ReadVariableOp_1�&batch_normalization_116/AssignNewValue�(batch_normalization_116/AssignNewValue_1�7batch_normalization_116/FusedBatchNormV3/ReadVariableOp�9batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_116/ReadVariableOp�(batch_normalization_116/ReadVariableOp_1�&batch_normalization_117/AssignNewValue�(batch_normalization_117/AssignNewValue_1�7batch_normalization_117/FusedBatchNormV3/ReadVariableOp�9batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_117/ReadVariableOp�(batch_normalization_117/ReadVariableOp_1�&batch_normalization_118/AssignNewValue�(batch_normalization_118/AssignNewValue_1�7batch_normalization_118/FusedBatchNormV3/ReadVariableOp�9batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_118/ReadVariableOp�(batch_normalization_118/ReadVariableOp_1�!conv2d_100/BiasAdd/ReadVariableOp� conv2d_100/Conv2D/ReadVariableOp�!conv2d_101/BiasAdd/ReadVariableOp� conv2d_101/Conv2D/ReadVariableOp� conv2d_96/BiasAdd/ReadVariableOp�conv2d_96/Conv2D/ReadVariableOp� conv2d_97/BiasAdd/ReadVariableOp�conv2d_97/Conv2D/ReadVariableOp� conv2d_98/BiasAdd/ReadVariableOp�conv2d_98/Conv2D/ReadVariableOp� conv2d_99/BiasAdd/ReadVariableOp�conv2d_99/Conv2D/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOp�
conv2d_96/Conv2D/ReadVariableOpReadVariableOp(conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_96/Conv2D/ReadVariableOp�
conv2d_96/Conv2DConv2Dinputs'conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_96/Conv2D�
 conv2d_96/BiasAdd/ReadVariableOpReadVariableOp)conv2d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_96/BiasAdd/ReadVariableOp�
conv2d_96/BiasAddBiasAddconv2d_96/Conv2D:output:0(conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_96/BiasAdd~
conv2d_96/ReluReluconv2d_96/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_96/Relu�
&batch_normalization_112/ReadVariableOpReadVariableOp/batch_normalization_112_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_112/ReadVariableOp�
(batch_normalization_112/ReadVariableOp_1ReadVariableOp1batch_normalization_112_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_112/ReadVariableOp_1�
7batch_normalization_112/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_112_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_112/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_112_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_112/FusedBatchNormV3FusedBatchNormV3conv2d_96/Relu:activations:0.batch_normalization_112/ReadVariableOp:value:00batch_normalization_112/ReadVariableOp_1:value:0?batch_normalization_112/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_112/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_112/FusedBatchNormV3�
&batch_normalization_112/AssignNewValueAssignVariableOp@batch_normalization_112_fusedbatchnormv3_readvariableop_resource5batch_normalization_112/FusedBatchNormV3:batch_mean:08^batch_normalization_112/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_112/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_112/AssignNewValue�
(batch_normalization_112/AssignNewValue_1AssignVariableOpBbatch_normalization_112_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_112/FusedBatchNormV3:batch_variance:0:^batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_112/AssignNewValue_1�
conv2d_97/Conv2D/ReadVariableOpReadVariableOp(conv2d_97_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_97/Conv2D/ReadVariableOp�
conv2d_97/Conv2DConv2D,batch_normalization_112/FusedBatchNormV3:y:0'conv2d_97/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_97/Conv2D�
 conv2d_97/BiasAdd/ReadVariableOpReadVariableOp)conv2d_97_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_97/BiasAdd/ReadVariableOp�
conv2d_97/BiasAddBiasAddconv2d_97/Conv2D:output:0(conv2d_97/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_97/BiasAdd~
conv2d_97/ReluReluconv2d_97/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_97/Relu�
&batch_normalization_113/ReadVariableOpReadVariableOp/batch_normalization_113_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_113/ReadVariableOp�
(batch_normalization_113/ReadVariableOp_1ReadVariableOp1batch_normalization_113_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_113/ReadVariableOp_1�
7batch_normalization_113/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_113_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_113/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_113_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_113/FusedBatchNormV3FusedBatchNormV3conv2d_97/Relu:activations:0.batch_normalization_113/ReadVariableOp:value:00batch_normalization_113/ReadVariableOp_1:value:0?batch_normalization_113/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_113/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_113/FusedBatchNormV3�
&batch_normalization_113/AssignNewValueAssignVariableOp@batch_normalization_113_fusedbatchnormv3_readvariableop_resource5batch_normalization_113/FusedBatchNormV3:batch_mean:08^batch_normalization_113/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_113/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_113/AssignNewValue�
(batch_normalization_113/AssignNewValue_1AssignVariableOpBbatch_normalization_113_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_113/FusedBatchNormV3:batch_variance:0:^batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_113/AssignNewValue_1�
max_pooling2d_48/MaxPoolMaxPool,batch_normalization_113/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_48/MaxPool�
conv2d_98/Conv2D/ReadVariableOpReadVariableOp(conv2d_98_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_98/Conv2D/ReadVariableOp�
conv2d_98/Conv2DConv2D!max_pooling2d_48/MaxPool:output:0'conv2d_98/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_98/Conv2D�
 conv2d_98/BiasAdd/ReadVariableOpReadVariableOp)conv2d_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_98/BiasAdd/ReadVariableOp�
conv2d_98/BiasAddBiasAddconv2d_98/Conv2D:output:0(conv2d_98/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_98/BiasAdd~
conv2d_98/ReluReluconv2d_98/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_98/Relu�
&batch_normalization_114/ReadVariableOpReadVariableOp/batch_normalization_114_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_114/ReadVariableOp�
(batch_normalization_114/ReadVariableOp_1ReadVariableOp1batch_normalization_114_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_114/ReadVariableOp_1�
7batch_normalization_114/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_114_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_114/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_114_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_114/FusedBatchNormV3FusedBatchNormV3conv2d_98/Relu:activations:0.batch_normalization_114/ReadVariableOp:value:00batch_normalization_114/ReadVariableOp_1:value:0?batch_normalization_114/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_114/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_114/FusedBatchNormV3�
&batch_normalization_114/AssignNewValueAssignVariableOp@batch_normalization_114_fusedbatchnormv3_readvariableop_resource5batch_normalization_114/FusedBatchNormV3:batch_mean:08^batch_normalization_114/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_114/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_114/AssignNewValue�
(batch_normalization_114/AssignNewValue_1AssignVariableOpBbatch_normalization_114_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_114/FusedBatchNormV3:batch_variance:0:^batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_114/AssignNewValue_1�
conv2d_99/Conv2D/ReadVariableOpReadVariableOp(conv2d_99_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_99/Conv2D/ReadVariableOp�
conv2d_99/Conv2DConv2D,batch_normalization_114/FusedBatchNormV3:y:0'conv2d_99/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_99/Conv2D�
 conv2d_99/BiasAdd/ReadVariableOpReadVariableOp)conv2d_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_99/BiasAdd/ReadVariableOp�
conv2d_99/BiasAddBiasAddconv2d_99/Conv2D:output:0(conv2d_99/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_99/BiasAdd~
conv2d_99/ReluReluconv2d_99/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_99/Relu�
&batch_normalization_115/ReadVariableOpReadVariableOp/batch_normalization_115_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_115/ReadVariableOp�
(batch_normalization_115/ReadVariableOp_1ReadVariableOp1batch_normalization_115_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_115/ReadVariableOp_1�
7batch_normalization_115/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_115_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_115/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_115_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_115/FusedBatchNormV3FusedBatchNormV3conv2d_99/Relu:activations:0.batch_normalization_115/ReadVariableOp:value:00batch_normalization_115/ReadVariableOp_1:value:0?batch_normalization_115/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_115/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_115/FusedBatchNormV3�
&batch_normalization_115/AssignNewValueAssignVariableOp@batch_normalization_115_fusedbatchnormv3_readvariableop_resource5batch_normalization_115/FusedBatchNormV3:batch_mean:08^batch_normalization_115/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_115/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_115/AssignNewValue�
(batch_normalization_115/AssignNewValue_1AssignVariableOpBbatch_normalization_115_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_115/FusedBatchNormV3:batch_variance:0:^batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_115/AssignNewValue_1�
max_pooling2d_49/MaxPoolMaxPool,batch_normalization_115/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_49/MaxPool�
 conv2d_100/Conv2D/ReadVariableOpReadVariableOp)conv2d_100_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02"
 conv2d_100/Conv2D/ReadVariableOp�
conv2d_100/Conv2DConv2D!max_pooling2d_49/MaxPool:output:0(conv2d_100/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_100/Conv2D�
!conv2d_100/BiasAdd/ReadVariableOpReadVariableOp*conv2d_100_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!conv2d_100/BiasAdd/ReadVariableOp�
conv2d_100/BiasAddBiasAddconv2d_100/Conv2D:output:0)conv2d_100/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_100/BiasAdd�
conv2d_100/ReluReluconv2d_100/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_100/Relu�
&batch_normalization_116/ReadVariableOpReadVariableOp/batch_normalization_116_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_116/ReadVariableOp�
(batch_normalization_116/ReadVariableOp_1ReadVariableOp1batch_normalization_116_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_116/ReadVariableOp_1�
7batch_normalization_116/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_116_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_116/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_116_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_116/FusedBatchNormV3FusedBatchNormV3conv2d_100/Relu:activations:0.batch_normalization_116/ReadVariableOp:value:00batch_normalization_116/ReadVariableOp_1:value:0?batch_normalization_116/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_116/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_116/FusedBatchNormV3�
&batch_normalization_116/AssignNewValueAssignVariableOp@batch_normalization_116_fusedbatchnormv3_readvariableop_resource5batch_normalization_116/FusedBatchNormV3:batch_mean:08^batch_normalization_116/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_116/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_116/AssignNewValue�
(batch_normalization_116/AssignNewValue_1AssignVariableOpBbatch_normalization_116_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_116/FusedBatchNormV3:batch_variance:0:^batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_116/AssignNewValue_1�
 conv2d_101/Conv2D/ReadVariableOpReadVariableOp)conv2d_101_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02"
 conv2d_101/Conv2D/ReadVariableOp�
conv2d_101/Conv2DConv2D,batch_normalization_116/FusedBatchNormV3:y:0(conv2d_101/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_101/Conv2D�
!conv2d_101/BiasAdd/ReadVariableOpReadVariableOp*conv2d_101_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!conv2d_101/BiasAdd/ReadVariableOp�
conv2d_101/BiasAddBiasAddconv2d_101/Conv2D:output:0)conv2d_101/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_101/BiasAdd�
conv2d_101/ReluReluconv2d_101/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_101/Relu�
&batch_normalization_117/ReadVariableOpReadVariableOp/batch_normalization_117_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_117/ReadVariableOp�
(batch_normalization_117/ReadVariableOp_1ReadVariableOp1batch_normalization_117_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_117/ReadVariableOp_1�
7batch_normalization_117/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_117_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_117/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_117_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_117/FusedBatchNormV3FusedBatchNormV3conv2d_101/Relu:activations:0.batch_normalization_117/ReadVariableOp:value:00batch_normalization_117/ReadVariableOp_1:value:0?batch_normalization_117/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_117/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_117/FusedBatchNormV3�
&batch_normalization_117/AssignNewValueAssignVariableOp@batch_normalization_117_fusedbatchnormv3_readvariableop_resource5batch_normalization_117/FusedBatchNormV3:batch_mean:08^batch_normalization_117/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_117/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_117/AssignNewValue�
(batch_normalization_117/AssignNewValue_1AssignVariableOpBbatch_normalization_117_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_117/FusedBatchNormV3:batch_variance:0:^batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_117/AssignNewValue_1�
max_pooling2d_50/MaxPoolMaxPool,batch_normalization_117/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_50/MaxPool�
&batch_normalization_118/ReadVariableOpReadVariableOp/batch_normalization_118_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_118/ReadVariableOp�
(batch_normalization_118/ReadVariableOp_1ReadVariableOp1batch_normalization_118_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_118/ReadVariableOp_1�
7batch_normalization_118/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_118_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_118/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_118_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_118/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_50/MaxPool:output:0.batch_normalization_118/ReadVariableOp:value:00batch_normalization_118/ReadVariableOp_1:value:0?batch_normalization_118/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_118/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_118/FusedBatchNormV3�
&batch_normalization_118/AssignNewValueAssignVariableOp@batch_normalization_118_fusedbatchnormv3_readvariableop_resource5batch_normalization_118/FusedBatchNormV3:batch_mean:08^batch_normalization_118/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_118/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_118/AssignNewValue�
(batch_normalization_118/AssignNewValue_1AssignVariableOpBbatch_normalization_118_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_118/FusedBatchNormV3:batch_variance:0:^batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_118/AssignNewValue_1u
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_16/Const�
flatten_16/ReshapeReshape,batch_normalization_118/FusedBatchNormV3:y:0flatten_16/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_16/Reshape�
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_17/MatMul/ReadVariableOp�
dense_17/MatMulMatMulflatten_16/Reshape:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_17/MatMul�
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_17/BiasAdd/ReadVariableOp�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_17/BiasAdd}
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_17/Sigmoid�
lambda_17/l2_normalize/SquareSquaredense_17/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
lambda_17/l2_normalize/Square�
,lambda_17/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,lambda_17/l2_normalize/Sum/reduction_indices�
lambda_17/l2_normalize/SumSum!lambda_17/l2_normalize/Square:y:05lambda_17/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
lambda_17/l2_normalize/Sum�
 lambda_17/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2"
 lambda_17/l2_normalize/Maximum/y�
lambda_17/l2_normalize/MaximumMaximum#lambda_17/l2_normalize/Sum:output:0)lambda_17/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2 
lambda_17/l2_normalize/Maximum�
lambda_17/l2_normalize/RsqrtRsqrt"lambda_17/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
lambda_17/l2_normalize/Rsqrt�
lambda_17/l2_normalizeMuldense_17/Sigmoid:y:0 lambda_17/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
lambda_17/l2_normalize�
IdentityIdentitylambda_17/l2_normalize:z:0'^batch_normalization_112/AssignNewValue)^batch_normalization_112/AssignNewValue_18^batch_normalization_112/FusedBatchNormV3/ReadVariableOp:^batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_112/ReadVariableOp)^batch_normalization_112/ReadVariableOp_1'^batch_normalization_113/AssignNewValue)^batch_normalization_113/AssignNewValue_18^batch_normalization_113/FusedBatchNormV3/ReadVariableOp:^batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_113/ReadVariableOp)^batch_normalization_113/ReadVariableOp_1'^batch_normalization_114/AssignNewValue)^batch_normalization_114/AssignNewValue_18^batch_normalization_114/FusedBatchNormV3/ReadVariableOp:^batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_114/ReadVariableOp)^batch_normalization_114/ReadVariableOp_1'^batch_normalization_115/AssignNewValue)^batch_normalization_115/AssignNewValue_18^batch_normalization_115/FusedBatchNormV3/ReadVariableOp:^batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_115/ReadVariableOp)^batch_normalization_115/ReadVariableOp_1'^batch_normalization_116/AssignNewValue)^batch_normalization_116/AssignNewValue_18^batch_normalization_116/FusedBatchNormV3/ReadVariableOp:^batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_116/ReadVariableOp)^batch_normalization_116/ReadVariableOp_1'^batch_normalization_117/AssignNewValue)^batch_normalization_117/AssignNewValue_18^batch_normalization_117/FusedBatchNormV3/ReadVariableOp:^batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_117/ReadVariableOp)^batch_normalization_117/ReadVariableOp_1'^batch_normalization_118/AssignNewValue)^batch_normalization_118/AssignNewValue_18^batch_normalization_118/FusedBatchNormV3/ReadVariableOp:^batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_118/ReadVariableOp)^batch_normalization_118/ReadVariableOp_1"^conv2d_100/BiasAdd/ReadVariableOp!^conv2d_100/Conv2D/ReadVariableOp"^conv2d_101/BiasAdd/ReadVariableOp!^conv2d_101/Conv2D/ReadVariableOp!^conv2d_96/BiasAdd/ReadVariableOp ^conv2d_96/Conv2D/ReadVariableOp!^conv2d_97/BiasAdd/ReadVariableOp ^conv2d_97/Conv2D/ReadVariableOp!^conv2d_98/BiasAdd/ReadVariableOp ^conv2d_98/Conv2D/ReadVariableOp!^conv2d_99/BiasAdd/ReadVariableOp ^conv2d_99/Conv2D/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2P
&batch_normalization_112/AssignNewValue&batch_normalization_112/AssignNewValue2T
(batch_normalization_112/AssignNewValue_1(batch_normalization_112/AssignNewValue_12r
7batch_normalization_112/FusedBatchNormV3/ReadVariableOp7batch_normalization_112/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_112/FusedBatchNormV3/ReadVariableOp_19batch_normalization_112/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_112/ReadVariableOp&batch_normalization_112/ReadVariableOp2T
(batch_normalization_112/ReadVariableOp_1(batch_normalization_112/ReadVariableOp_12P
&batch_normalization_113/AssignNewValue&batch_normalization_113/AssignNewValue2T
(batch_normalization_113/AssignNewValue_1(batch_normalization_113/AssignNewValue_12r
7batch_normalization_113/FusedBatchNormV3/ReadVariableOp7batch_normalization_113/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_113/FusedBatchNormV3/ReadVariableOp_19batch_normalization_113/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_113/ReadVariableOp&batch_normalization_113/ReadVariableOp2T
(batch_normalization_113/ReadVariableOp_1(batch_normalization_113/ReadVariableOp_12P
&batch_normalization_114/AssignNewValue&batch_normalization_114/AssignNewValue2T
(batch_normalization_114/AssignNewValue_1(batch_normalization_114/AssignNewValue_12r
7batch_normalization_114/FusedBatchNormV3/ReadVariableOp7batch_normalization_114/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_114/FusedBatchNormV3/ReadVariableOp_19batch_normalization_114/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_114/ReadVariableOp&batch_normalization_114/ReadVariableOp2T
(batch_normalization_114/ReadVariableOp_1(batch_normalization_114/ReadVariableOp_12P
&batch_normalization_115/AssignNewValue&batch_normalization_115/AssignNewValue2T
(batch_normalization_115/AssignNewValue_1(batch_normalization_115/AssignNewValue_12r
7batch_normalization_115/FusedBatchNormV3/ReadVariableOp7batch_normalization_115/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_115/FusedBatchNormV3/ReadVariableOp_19batch_normalization_115/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_115/ReadVariableOp&batch_normalization_115/ReadVariableOp2T
(batch_normalization_115/ReadVariableOp_1(batch_normalization_115/ReadVariableOp_12P
&batch_normalization_116/AssignNewValue&batch_normalization_116/AssignNewValue2T
(batch_normalization_116/AssignNewValue_1(batch_normalization_116/AssignNewValue_12r
7batch_normalization_116/FusedBatchNormV3/ReadVariableOp7batch_normalization_116/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_116/FusedBatchNormV3/ReadVariableOp_19batch_normalization_116/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_116/ReadVariableOp&batch_normalization_116/ReadVariableOp2T
(batch_normalization_116/ReadVariableOp_1(batch_normalization_116/ReadVariableOp_12P
&batch_normalization_117/AssignNewValue&batch_normalization_117/AssignNewValue2T
(batch_normalization_117/AssignNewValue_1(batch_normalization_117/AssignNewValue_12r
7batch_normalization_117/FusedBatchNormV3/ReadVariableOp7batch_normalization_117/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_117/FusedBatchNormV3/ReadVariableOp_19batch_normalization_117/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_117/ReadVariableOp&batch_normalization_117/ReadVariableOp2T
(batch_normalization_117/ReadVariableOp_1(batch_normalization_117/ReadVariableOp_12P
&batch_normalization_118/AssignNewValue&batch_normalization_118/AssignNewValue2T
(batch_normalization_118/AssignNewValue_1(batch_normalization_118/AssignNewValue_12r
7batch_normalization_118/FusedBatchNormV3/ReadVariableOp7batch_normalization_118/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_118/FusedBatchNormV3/ReadVariableOp_19batch_normalization_118/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_118/ReadVariableOp&batch_normalization_118/ReadVariableOp2T
(batch_normalization_118/ReadVariableOp_1(batch_normalization_118/ReadVariableOp_12F
!conv2d_100/BiasAdd/ReadVariableOp!conv2d_100/BiasAdd/ReadVariableOp2D
 conv2d_100/Conv2D/ReadVariableOp conv2d_100/Conv2D/ReadVariableOp2F
!conv2d_101/BiasAdd/ReadVariableOp!conv2d_101/BiasAdd/ReadVariableOp2D
 conv2d_101/Conv2D/ReadVariableOp conv2d_101/Conv2D/ReadVariableOp2D
 conv2d_96/BiasAdd/ReadVariableOp conv2d_96/BiasAdd/ReadVariableOp2B
conv2d_96/Conv2D/ReadVariableOpconv2d_96/Conv2D/ReadVariableOp2D
 conv2d_97/BiasAdd/ReadVariableOp conv2d_97/BiasAdd/ReadVariableOp2B
conv2d_97/Conv2D/ReadVariableOpconv2d_97/Conv2D/ReadVariableOp2D
 conv2d_98/BiasAdd/ReadVariableOp conv2d_98/BiasAdd/ReadVariableOp2B
conv2d_98/Conv2D/ReadVariableOpconv2d_98/Conv2D/ReadVariableOp2D
 conv2d_99/BiasAdd/ReadVariableOp conv2d_99/BiasAdd/ReadVariableOp2B
conv2d_99/Conv2D/ReadVariableOpconv2d_99/Conv2D/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�

�
F__inference_conv2d_100_layer_call_and_return_conditional_losses_227734

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_230027

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_lambda_17_layer_call_fn_230257

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lambda_17_layer_call_and_return_conditional_losses_2280492
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_conv2d_98_layer_call_and_return_conditional_losses_227533

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_226866

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_227367

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_226767

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_227769

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�j
�
I__inference_sequential_16_layer_call_and_return_conditional_losses_228291

inputs
conv2d_96_228187
conv2d_96_228189"
batch_normalization_112_228192"
batch_normalization_112_228194"
batch_normalization_112_228196"
batch_normalization_112_228198
conv2d_97_228201
conv2d_97_228203"
batch_normalization_113_228206"
batch_normalization_113_228208"
batch_normalization_113_228210"
batch_normalization_113_228212
conv2d_98_228216
conv2d_98_228218"
batch_normalization_114_228221"
batch_normalization_114_228223"
batch_normalization_114_228225"
batch_normalization_114_228227
conv2d_99_228230
conv2d_99_228232"
batch_normalization_115_228235"
batch_normalization_115_228237"
batch_normalization_115_228239"
batch_normalization_115_228241
conv2d_100_228245
conv2d_100_228247"
batch_normalization_116_228250"
batch_normalization_116_228252"
batch_normalization_116_228254"
batch_normalization_116_228256
conv2d_101_228259
conv2d_101_228261"
batch_normalization_117_228264"
batch_normalization_117_228266"
batch_normalization_117_228268"
batch_normalization_117_228270"
batch_normalization_118_228274"
batch_normalization_118_228276"
batch_normalization_118_228278"
batch_normalization_118_228280
dense_17_228284
dense_17_228286
identity��/batch_normalization_112/StatefulPartitionedCall�/batch_normalization_113/StatefulPartitionedCall�/batch_normalization_114/StatefulPartitionedCall�/batch_normalization_115/StatefulPartitionedCall�/batch_normalization_116/StatefulPartitionedCall�/batch_normalization_117/StatefulPartitionedCall�/batch_normalization_118/StatefulPartitionedCall�"conv2d_100/StatefulPartitionedCall�"conv2d_101/StatefulPartitionedCall�!conv2d_96/StatefulPartitionedCall�!conv2d_97/StatefulPartitionedCall�!conv2d_98/StatefulPartitionedCall�!conv2d_99/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_96_228187conv2d_96_228189*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_2273322#
!conv2d_96/StatefulPartitionedCall�
/batch_normalization_112/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0batch_normalization_112_228192batch_normalization_112_228194batch_normalization_112_228196batch_normalization_112_228198*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_22736721
/batch_normalization_112/StatefulPartitionedCall�
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_112/StatefulPartitionedCall:output:0conv2d_97_228201conv2d_97_228203*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_2274322#
!conv2d_97/StatefulPartitionedCall�
/batch_normalization_113/StatefulPartitionedCallStatefulPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0batch_normalization_113_228206batch_normalization_113_228208batch_normalization_113_228210batch_normalization_113_228212*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_22746721
/batch_normalization_113/StatefulPartitionedCall�
 max_pooling2d_48/PartitionedCallPartitionedCall8batch_normalization_113/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2267672"
 max_pooling2d_48/PartitionedCall�
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0conv2d_98_228216conv2d_98_228218*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_98_layer_call_and_return_conditional_losses_2275332#
!conv2d_98/StatefulPartitionedCall�
/batch_normalization_114/StatefulPartitionedCallStatefulPartitionedCall*conv2d_98/StatefulPartitionedCall:output:0batch_normalization_114_228221batch_normalization_114_228223batch_normalization_114_228225batch_normalization_114_228227*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_22756821
/batch_normalization_114/StatefulPartitionedCall�
!conv2d_99/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_114/StatefulPartitionedCall:output:0conv2d_99_228230conv2d_99_228232*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_99_layer_call_and_return_conditional_losses_2276332#
!conv2d_99/StatefulPartitionedCall�
/batch_normalization_115/StatefulPartitionedCallStatefulPartitionedCall*conv2d_99/StatefulPartitionedCall:output:0batch_normalization_115_228235batch_normalization_115_228237batch_normalization_115_228239batch_normalization_115_228241*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_22766821
/batch_normalization_115/StatefulPartitionedCall�
 max_pooling2d_49/PartitionedCallPartitionedCall8batch_normalization_115/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2269872"
 max_pooling2d_49/PartitionedCall�
"conv2d_100/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_100_228245conv2d_100_228247*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_100_layer_call_and_return_conditional_losses_2277342$
"conv2d_100/StatefulPartitionedCall�
/batch_normalization_116/StatefulPartitionedCallStatefulPartitionedCall+conv2d_100/StatefulPartitionedCall:output:0batch_normalization_116_228250batch_normalization_116_228252batch_normalization_116_228254batch_normalization_116_228256*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_22776921
/batch_normalization_116/StatefulPartitionedCall�
"conv2d_101/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_116/StatefulPartitionedCall:output:0conv2d_101_228259conv2d_101_228261*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_101_layer_call_and_return_conditional_losses_2278342$
"conv2d_101/StatefulPartitionedCall�
/batch_normalization_117/StatefulPartitionedCallStatefulPartitionedCall+conv2d_101/StatefulPartitionedCall:output:0batch_normalization_117_228264batch_normalization_117_228266batch_normalization_117_228268batch_normalization_117_228270*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_22786921
/batch_normalization_117/StatefulPartitionedCall�
 max_pooling2d_50/PartitionedCallPartitionedCall8batch_normalization_117/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2272072"
 max_pooling2d_50/PartitionedCall�
/batch_normalization_118/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0batch_normalization_118_228274batch_normalization_118_228276batch_normalization_118_228278batch_normalization_118_228280*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_22794321
/batch_normalization_118/StatefulPartitionedCall�
flatten_16/PartitionedCallPartitionedCall8batch_normalization_118/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_2280032
flatten_16/PartitionedCall�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_17_228284dense_17_228286*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_2280222"
 dense_17/StatefulPartitionedCall�
lambda_17/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lambda_17_layer_call_and_return_conditional_losses_2280492
lambda_17/PartitionedCall�
IdentityIdentity"lambda_17/PartitionedCall:output:00^batch_normalization_112/StatefulPartitionedCall0^batch_normalization_113/StatefulPartitionedCall0^batch_normalization_114/StatefulPartitionedCall0^batch_normalization_115/StatefulPartitionedCall0^batch_normalization_116/StatefulPartitionedCall0^batch_normalization_117/StatefulPartitionedCall0^batch_normalization_118/StatefulPartitionedCall#^conv2d_100/StatefulPartitionedCall#^conv2d_101/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall"^conv2d_98/StatefulPartitionedCall"^conv2d_99/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2b
/batch_normalization_112/StatefulPartitionedCall/batch_normalization_112/StatefulPartitionedCall2b
/batch_normalization_113/StatefulPartitionedCall/batch_normalization_113/StatefulPartitionedCall2b
/batch_normalization_114/StatefulPartitionedCall/batch_normalization_114/StatefulPartitionedCall2b
/batch_normalization_115/StatefulPartitionedCall/batch_normalization_115/StatefulPartitionedCall2b
/batch_normalization_116/StatefulPartitionedCall/batch_normalization_116/StatefulPartitionedCall2b
/batch_normalization_117/StatefulPartitionedCall/batch_normalization_117/StatefulPartitionedCall2b
/batch_normalization_118/StatefulPartitionedCall/batch_normalization_118/StatefulPartitionedCall2H
"conv2d_100/StatefulPartitionedCall"conv2d_100/StatefulPartitionedCall2H
"conv2d_101/StatefulPartitionedCall"conv2d_101/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall2F
!conv2d_99/StatefulPartitionedCall!conv2d_99/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�

*__inference_conv2d_96_layer_call_fn_229203

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_2273322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_115_layer_call_fn_229775

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_2276862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_114_layer_call_fn_229563

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_2268662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
b
F__inference_flatten_16_layer_call_and_return_conditional_losses_228003

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229241

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230173

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_227586

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_227869

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_116_layer_call_fn_229846

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_2270552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_117_layer_call_fn_230058

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_2278692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
~
)__inference_dense_17_layer_call_fn_230230

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_2280222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
D__inference_dense_17_layer_call_and_return_conditional_losses_230221

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_118_layer_call_fn_230135

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_2273062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_227686

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229287

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230091

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�[
�
__inference__traced_save_230411
file_prefix/
+savev2_conv2d_96_kernel_read_readvariableop-
)savev2_conv2d_96_bias_read_readvariableop<
8savev2_batch_normalization_112_gamma_read_readvariableop;
7savev2_batch_normalization_112_beta_read_readvariableopB
>savev2_batch_normalization_112_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_112_moving_variance_read_readvariableop/
+savev2_conv2d_97_kernel_read_readvariableop-
)savev2_conv2d_97_bias_read_readvariableop<
8savev2_batch_normalization_113_gamma_read_readvariableop;
7savev2_batch_normalization_113_beta_read_readvariableopB
>savev2_batch_normalization_113_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_113_moving_variance_read_readvariableop/
+savev2_conv2d_98_kernel_read_readvariableop-
)savev2_conv2d_98_bias_read_readvariableop<
8savev2_batch_normalization_114_gamma_read_readvariableop;
7savev2_batch_normalization_114_beta_read_readvariableopB
>savev2_batch_normalization_114_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_114_moving_variance_read_readvariableop/
+savev2_conv2d_99_kernel_read_readvariableop-
)savev2_conv2d_99_bias_read_readvariableop<
8savev2_batch_normalization_115_gamma_read_readvariableop;
7savev2_batch_normalization_115_beta_read_readvariableopB
>savev2_batch_normalization_115_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_115_moving_variance_read_readvariableop0
,savev2_conv2d_100_kernel_read_readvariableop.
*savev2_conv2d_100_bias_read_readvariableop<
8savev2_batch_normalization_116_gamma_read_readvariableop;
7savev2_batch_normalization_116_beta_read_readvariableopB
>savev2_batch_normalization_116_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_116_moving_variance_read_readvariableop0
,savev2_conv2d_101_kernel_read_readvariableop.
*savev2_conv2d_101_bias_read_readvariableop<
8savev2_batch_normalization_117_gamma_read_readvariableop;
7savev2_batch_normalization_117_beta_read_readvariableopB
>savev2_batch_normalization_117_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_117_moving_variance_read_readvariableop<
8savev2_batch_normalization_118_gamma_read_readvariableop;
7savev2_batch_normalization_118_beta_read_readvariableopB
>savev2_batch_normalization_118_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_118_moving_variance_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_96_kernel_read_readvariableop)savev2_conv2d_96_bias_read_readvariableop8savev2_batch_normalization_112_gamma_read_readvariableop7savev2_batch_normalization_112_beta_read_readvariableop>savev2_batch_normalization_112_moving_mean_read_readvariableopBsavev2_batch_normalization_112_moving_variance_read_readvariableop+savev2_conv2d_97_kernel_read_readvariableop)savev2_conv2d_97_bias_read_readvariableop8savev2_batch_normalization_113_gamma_read_readvariableop7savev2_batch_normalization_113_beta_read_readvariableop>savev2_batch_normalization_113_moving_mean_read_readvariableopBsavev2_batch_normalization_113_moving_variance_read_readvariableop+savev2_conv2d_98_kernel_read_readvariableop)savev2_conv2d_98_bias_read_readvariableop8savev2_batch_normalization_114_gamma_read_readvariableop7savev2_batch_normalization_114_beta_read_readvariableop>savev2_batch_normalization_114_moving_mean_read_readvariableopBsavev2_batch_normalization_114_moving_variance_read_readvariableop+savev2_conv2d_99_kernel_read_readvariableop)savev2_conv2d_99_bias_read_readvariableop8savev2_batch_normalization_115_gamma_read_readvariableop7savev2_batch_normalization_115_beta_read_readvariableop>savev2_batch_normalization_115_moving_mean_read_readvariableopBsavev2_batch_normalization_115_moving_variance_read_readvariableop,savev2_conv2d_100_kernel_read_readvariableop*savev2_conv2d_100_bias_read_readvariableop8savev2_batch_normalization_116_gamma_read_readvariableop7savev2_batch_normalization_116_beta_read_readvariableop>savev2_batch_normalization_116_moving_mean_read_readvariableopBsavev2_batch_normalization_116_moving_variance_read_readvariableop,savev2_conv2d_101_kernel_read_readvariableop*savev2_conv2d_101_bias_read_readvariableop8savev2_batch_normalization_117_gamma_read_readvariableop7savev2_batch_normalization_117_beta_read_readvariableop>savev2_batch_normalization_117_moving_mean_read_readvariableopBsavev2_batch_normalization_117_moving_variance_read_readvariableop8savev2_batch_normalization_118_gamma_read_readvariableop7savev2_batch_normalization_118_beta_read_readvariableop>savev2_batch_normalization_118_moving_mean_read_readvariableopBsavev2_batch_normalization_118_moving_variance_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : :  : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:@�:�:�:�:�:�:��:�:�:�:�:�:�:�:�:�:
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:! 

_output_shapes	
:�:!!

_output_shapes	
:�:!"

_output_shapes	
:�:!#

_output_shapes	
:�:!$

_output_shapes	
:�:!%

_output_shapes	
:�:!&

_output_shapes	
:�:!'

_output_shapes	
:�:!(

_output_shapes	
:�:&)"
 
_output_shapes
:
��:!*

_output_shapes	
:�:+

_output_shapes
: 
�
�
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_227568

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
G
+__inference_flatten_16_layer_call_fn_230210

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_2280032
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_conv2d_97_layer_call_and_return_conditional_losses_229342

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_118_layer_call_fn_230186

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_2279432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229519

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_227159

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_112_layer_call_fn_229331

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_2266462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
.__inference_sequential_16_layer_call_fn_228574
conv2d_96_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_96_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_2284872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_96_input
�
�
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229371

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229583

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_226939

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
+__inference_conv2d_100_layer_call_fn_229795

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_100_layer_call_and_return_conditional_losses_2277342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_96_layer_call_and_return_conditional_losses_229194

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_227207

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229685

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
F__inference_conv2d_100_layer_call_and_return_conditional_losses_229786

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

a
E__inference_lambda_17_layer_call_and_return_conditional_losses_230241

inputs
identityo
l2_normalize/SquareSquareinputs*
T0*(
_output_shapes
:����������2
l2_normalize/Square�
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2$
"l2_normalize/Sum/reduction_indices�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
l2_normalize/Sumu
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2
l2_normalize/Maximum/y�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2
l2_normalize/Maximum}
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
l2_normalize/Rsqrtv
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
l2_normalizee
IdentityIdentityl2_normalize:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_114_layer_call_fn_229627

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_2275862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229833

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

*__inference_conv2d_99_layer_call_fn_229647

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_99_layer_call_and_return_conditional_losses_2276332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_115_layer_call_fn_229762

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_2276682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_227961

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_227306

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_226646

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229749

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_97_layer_call_and_return_conditional_losses_227432

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_230045

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_113_layer_call_fn_229466

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_2274672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�k
�
I__inference_sequential_16_layer_call_and_return_conditional_losses_228487

inputs
conv2d_96_228383
conv2d_96_228385"
batch_normalization_112_228388"
batch_normalization_112_228390"
batch_normalization_112_228392"
batch_normalization_112_228394
conv2d_97_228397
conv2d_97_228399"
batch_normalization_113_228402"
batch_normalization_113_228404"
batch_normalization_113_228406"
batch_normalization_113_228408
conv2d_98_228412
conv2d_98_228414"
batch_normalization_114_228417"
batch_normalization_114_228419"
batch_normalization_114_228421"
batch_normalization_114_228423
conv2d_99_228426
conv2d_99_228428"
batch_normalization_115_228431"
batch_normalization_115_228433"
batch_normalization_115_228435"
batch_normalization_115_228437
conv2d_100_228441
conv2d_100_228443"
batch_normalization_116_228446"
batch_normalization_116_228448"
batch_normalization_116_228450"
batch_normalization_116_228452
conv2d_101_228455
conv2d_101_228457"
batch_normalization_117_228460"
batch_normalization_117_228462"
batch_normalization_117_228464"
batch_normalization_117_228466"
batch_normalization_118_228470"
batch_normalization_118_228472"
batch_normalization_118_228474"
batch_normalization_118_228476
dense_17_228480
dense_17_228482
identity��/batch_normalization_112/StatefulPartitionedCall�/batch_normalization_113/StatefulPartitionedCall�/batch_normalization_114/StatefulPartitionedCall�/batch_normalization_115/StatefulPartitionedCall�/batch_normalization_116/StatefulPartitionedCall�/batch_normalization_117/StatefulPartitionedCall�/batch_normalization_118/StatefulPartitionedCall�"conv2d_100/StatefulPartitionedCall�"conv2d_101/StatefulPartitionedCall�!conv2d_96/StatefulPartitionedCall�!conv2d_97/StatefulPartitionedCall�!conv2d_98/StatefulPartitionedCall�!conv2d_99/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_96_228383conv2d_96_228385*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_2273322#
!conv2d_96/StatefulPartitionedCall�
/batch_normalization_112/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0batch_normalization_112_228388batch_normalization_112_228390batch_normalization_112_228392batch_normalization_112_228394*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_22738521
/batch_normalization_112/StatefulPartitionedCall�
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_112/StatefulPartitionedCall:output:0conv2d_97_228397conv2d_97_228399*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_2274322#
!conv2d_97/StatefulPartitionedCall�
/batch_normalization_113/StatefulPartitionedCallStatefulPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0batch_normalization_113_228402batch_normalization_113_228404batch_normalization_113_228406batch_normalization_113_228408*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_22748521
/batch_normalization_113/StatefulPartitionedCall�
 max_pooling2d_48/PartitionedCallPartitionedCall8batch_normalization_113/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2267672"
 max_pooling2d_48/PartitionedCall�
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0conv2d_98_228412conv2d_98_228414*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_98_layer_call_and_return_conditional_losses_2275332#
!conv2d_98/StatefulPartitionedCall�
/batch_normalization_114/StatefulPartitionedCallStatefulPartitionedCall*conv2d_98/StatefulPartitionedCall:output:0batch_normalization_114_228417batch_normalization_114_228419batch_normalization_114_228421batch_normalization_114_228423*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_22758621
/batch_normalization_114/StatefulPartitionedCall�
!conv2d_99/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_114/StatefulPartitionedCall:output:0conv2d_99_228426conv2d_99_228428*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_99_layer_call_and_return_conditional_losses_2276332#
!conv2d_99/StatefulPartitionedCall�
/batch_normalization_115/StatefulPartitionedCallStatefulPartitionedCall*conv2d_99/StatefulPartitionedCall:output:0batch_normalization_115_228431batch_normalization_115_228433batch_normalization_115_228435batch_normalization_115_228437*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_22768621
/batch_normalization_115/StatefulPartitionedCall�
 max_pooling2d_49/PartitionedCallPartitionedCall8batch_normalization_115/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2269872"
 max_pooling2d_49/PartitionedCall�
"conv2d_100/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_100_228441conv2d_100_228443*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_100_layer_call_and_return_conditional_losses_2277342$
"conv2d_100/StatefulPartitionedCall�
/batch_normalization_116/StatefulPartitionedCallStatefulPartitionedCall+conv2d_100/StatefulPartitionedCall:output:0batch_normalization_116_228446batch_normalization_116_228448batch_normalization_116_228450batch_normalization_116_228452*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_22778721
/batch_normalization_116/StatefulPartitionedCall�
"conv2d_101/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_116/StatefulPartitionedCall:output:0conv2d_101_228455conv2d_101_228457*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_101_layer_call_and_return_conditional_losses_2278342$
"conv2d_101/StatefulPartitionedCall�
/batch_normalization_117/StatefulPartitionedCallStatefulPartitionedCall+conv2d_101/StatefulPartitionedCall:output:0batch_normalization_117_228460batch_normalization_117_228462batch_normalization_117_228464batch_normalization_117_228466*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_22788721
/batch_normalization_117/StatefulPartitionedCall�
 max_pooling2d_50/PartitionedCallPartitionedCall8batch_normalization_117/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2272072"
 max_pooling2d_50/PartitionedCall�
/batch_normalization_118/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0batch_normalization_118_228470batch_normalization_118_228472batch_normalization_118_228474batch_normalization_118_228476*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_22796121
/batch_normalization_118/StatefulPartitionedCall�
flatten_16/PartitionedCallPartitionedCall8batch_normalization_118/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_2280032
flatten_16/PartitionedCall�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_17_228480dense_17_228482*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_2280222"
 dense_17/StatefulPartitionedCall�
lambda_17/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lambda_17_layer_call_and_return_conditional_losses_2280602
lambda_17/PartitionedCall�
IdentityIdentity"lambda_17/PartitionedCall:output:00^batch_normalization_112/StatefulPartitionedCall0^batch_normalization_113/StatefulPartitionedCall0^batch_normalization_114/StatefulPartitionedCall0^batch_normalization_115/StatefulPartitionedCall0^batch_normalization_116/StatefulPartitionedCall0^batch_normalization_117/StatefulPartitionedCall0^batch_normalization_118/StatefulPartitionedCall#^conv2d_100/StatefulPartitionedCall#^conv2d_101/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall"^conv2d_98/StatefulPartitionedCall"^conv2d_99/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2b
/batch_normalization_112/StatefulPartitionedCall/batch_normalization_112/StatefulPartitionedCall2b
/batch_normalization_113/StatefulPartitionedCall/batch_normalization_113/StatefulPartitionedCall2b
/batch_normalization_114/StatefulPartitionedCall/batch_normalization_114/StatefulPartitionedCall2b
/batch_normalization_115/StatefulPartitionedCall/batch_normalization_115/StatefulPartitionedCall2b
/batch_normalization_116/StatefulPartitionedCall/batch_normalization_116/StatefulPartitionedCall2b
/batch_normalization_117/StatefulPartitionedCall/batch_normalization_117/StatefulPartitionedCall2b
/batch_normalization_118/StatefulPartitionedCall/batch_normalization_118/StatefulPartitionedCall2H
"conv2d_100/StatefulPartitionedCall"conv2d_100/StatefulPartitionedCall2H
"conv2d_101/StatefulPartitionedCall"conv2d_101/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall2F
!conv2d_99/StatefulPartitionedCall!conv2d_99/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_114_layer_call_fn_229550

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_2268352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_227385

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_113_layer_call_fn_229415

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_2267502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_226750

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_227055

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

a
E__inference_lambda_17_layer_call_and_return_conditional_losses_228060

inputs
identityo
l2_normalize/SquareSquareinputs*
T0*(
_output_shapes
:����������2
l2_normalize/Square�
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2$
"l2_normalize/Sum/reduction_indices�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
l2_normalize/Sumu
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2
l2_normalize/Maximum/y�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2
l2_normalize/Maximum}
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
l2_normalize/Rsqrtv
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
l2_normalizee
IdentityIdentityl2_normalize:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

*__inference_conv2d_97_layer_call_fn_229351

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_2274322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�

�
F__inference_conv2d_101_layer_call_and_return_conditional_losses_229934

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_117_layer_call_fn_230007

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_2271902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_117_layer_call_fn_229994

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_2271592
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_113_layer_call_fn_229479

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_2274852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_115_layer_call_fn_229711

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_2269702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_112_layer_call_fn_229267

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_2273852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_227190

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229667

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_96_layer_call_and_return_conditional_losses_227332

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������   2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_227887

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

*__inference_conv2d_98_layer_call_fn_229499

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_98_layer_call_and_return_conditional_losses_2275332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
.__inference_sequential_16_layer_call_fn_229183

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_2284872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_49_layer_call_fn_226993

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2269872
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_227485

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
F
*__inference_lambda_17_layer_call_fn_230262

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lambda_17_layer_call_and_return_conditional_losses_2280602
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_118_layer_call_fn_230122

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_2272752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_115_layer_call_fn_229698

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_2269392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
F__inference_conv2d_101_layer_call_and_return_conditional_losses_227834

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_229981

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

a
E__inference_lambda_17_layer_call_and_return_conditional_losses_230252

inputs
identityo
l2_normalize/SquareSquareinputs*
T0*(
_output_shapes
:����������2
l2_normalize/Square�
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2$
"l2_normalize/Sum/reduction_indices�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
l2_normalize/Sumu
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2
l2_normalize/Maximum/y�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2
l2_normalize/Maximum}
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
l2_normalize/Rsqrtv
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
l2_normalizee
IdentityIdentityl2_normalize:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230109

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229389

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_114_layer_call_fn_229614

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_2275682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229879

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_conv2d_99_layer_call_and_return_conditional_losses_229638

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_226719

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229435

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229537

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
E__inference_conv2d_98_layer_call_and_return_conditional_losses_229490

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229815

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_227275

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_230547
file_prefix%
!assignvariableop_conv2d_96_kernel%
!assignvariableop_1_conv2d_96_bias4
0assignvariableop_2_batch_normalization_112_gamma3
/assignvariableop_3_batch_normalization_112_beta:
6assignvariableop_4_batch_normalization_112_moving_mean>
:assignvariableop_5_batch_normalization_112_moving_variance'
#assignvariableop_6_conv2d_97_kernel%
!assignvariableop_7_conv2d_97_bias4
0assignvariableop_8_batch_normalization_113_gamma3
/assignvariableop_9_batch_normalization_113_beta;
7assignvariableop_10_batch_normalization_113_moving_mean?
;assignvariableop_11_batch_normalization_113_moving_variance(
$assignvariableop_12_conv2d_98_kernel&
"assignvariableop_13_conv2d_98_bias5
1assignvariableop_14_batch_normalization_114_gamma4
0assignvariableop_15_batch_normalization_114_beta;
7assignvariableop_16_batch_normalization_114_moving_mean?
;assignvariableop_17_batch_normalization_114_moving_variance(
$assignvariableop_18_conv2d_99_kernel&
"assignvariableop_19_conv2d_99_bias5
1assignvariableop_20_batch_normalization_115_gamma4
0assignvariableop_21_batch_normalization_115_beta;
7assignvariableop_22_batch_normalization_115_moving_mean?
;assignvariableop_23_batch_normalization_115_moving_variance)
%assignvariableop_24_conv2d_100_kernel'
#assignvariableop_25_conv2d_100_bias5
1assignvariableop_26_batch_normalization_116_gamma4
0assignvariableop_27_batch_normalization_116_beta;
7assignvariableop_28_batch_normalization_116_moving_mean?
;assignvariableop_29_batch_normalization_116_moving_variance)
%assignvariableop_30_conv2d_101_kernel'
#assignvariableop_31_conv2d_101_bias5
1assignvariableop_32_batch_normalization_117_gamma4
0assignvariableop_33_batch_normalization_117_beta;
7assignvariableop_34_batch_normalization_117_moving_mean?
;assignvariableop_35_batch_normalization_117_moving_variance5
1assignvariableop_36_batch_normalization_118_gamma4
0assignvariableop_37_batch_normalization_118_beta;
7assignvariableop_38_batch_normalization_118_moving_mean?
;assignvariableop_39_batch_normalization_118_moving_variance'
#assignvariableop_40_dense_17_kernel%
!assignvariableop_41_dense_17_bias
identity_43��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_96_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_96_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_batch_normalization_112_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_batch_normalization_112_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_112_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp:assignvariableop_5_batch_normalization_112_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_97_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_97_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_batch_normalization_113_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_113_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_113_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp;assignvariableop_11_batch_normalization_113_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_98_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_98_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_batch_normalization_114_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_114_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp7assignvariableop_16_batch_normalization_114_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp;assignvariableop_17_batch_normalization_114_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_99_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_99_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp1assignvariableop_20_batch_normalization_115_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_115_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp7assignvariableop_22_batch_normalization_115_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp;assignvariableop_23_batch_normalization_115_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp%assignvariableop_24_conv2d_100_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp#assignvariableop_25_conv2d_100_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp1assignvariableop_26_batch_normalization_116_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp0assignvariableop_27_batch_normalization_116_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp7assignvariableop_28_batch_normalization_116_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp;assignvariableop_29_batch_normalization_116_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp%assignvariableop_30_conv2d_101_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp#assignvariableop_31_conv2d_101_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp1assignvariableop_32_batch_normalization_117_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp0assignvariableop_33_batch_normalization_117_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp7assignvariableop_34_batch_normalization_117_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp;assignvariableop_35_batch_normalization_117_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp1assignvariableop_36_batch_normalization_118_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp0assignvariableop_37_batch_normalization_118_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp7assignvariableop_38_batch_normalization_118_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp;assignvariableop_39_batch_normalization_118_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp#assignvariableop_40_dense_17_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp!assignvariableop_41_dense_17_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_419
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_42�
Identity_43IdentityIdentity_42:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_43"#
identity_43Identity_43:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230155

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_229963

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�"
I__inference_sequential_16_layer_call_and_return_conditional_losses_229005

inputs,
(conv2d_96_conv2d_readvariableop_resource-
)conv2d_96_biasadd_readvariableop_resource3
/batch_normalization_112_readvariableop_resource5
1batch_normalization_112_readvariableop_1_resourceD
@batch_normalization_112_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_112_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_97_conv2d_readvariableop_resource-
)conv2d_97_biasadd_readvariableop_resource3
/batch_normalization_113_readvariableop_resource5
1batch_normalization_113_readvariableop_1_resourceD
@batch_normalization_113_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_113_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_98_conv2d_readvariableop_resource-
)conv2d_98_biasadd_readvariableop_resource3
/batch_normalization_114_readvariableop_resource5
1batch_normalization_114_readvariableop_1_resourceD
@batch_normalization_114_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_114_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_99_conv2d_readvariableop_resource-
)conv2d_99_biasadd_readvariableop_resource3
/batch_normalization_115_readvariableop_resource5
1batch_normalization_115_readvariableop_1_resourceD
@batch_normalization_115_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_115_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_100_conv2d_readvariableop_resource.
*conv2d_100_biasadd_readvariableop_resource3
/batch_normalization_116_readvariableop_resource5
1batch_normalization_116_readvariableop_1_resourceD
@batch_normalization_116_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_116_fusedbatchnormv3_readvariableop_1_resource-
)conv2d_101_conv2d_readvariableop_resource.
*conv2d_101_biasadd_readvariableop_resource3
/batch_normalization_117_readvariableop_resource5
1batch_normalization_117_readvariableop_1_resourceD
@batch_normalization_117_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_117_fusedbatchnormv3_readvariableop_1_resource3
/batch_normalization_118_readvariableop_resource5
1batch_normalization_118_readvariableop_1_resourceD
@batch_normalization_118_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_118_fusedbatchnormv3_readvariableop_1_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity��7batch_normalization_112/FusedBatchNormV3/ReadVariableOp�9batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_112/ReadVariableOp�(batch_normalization_112/ReadVariableOp_1�7batch_normalization_113/FusedBatchNormV3/ReadVariableOp�9batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_113/ReadVariableOp�(batch_normalization_113/ReadVariableOp_1�7batch_normalization_114/FusedBatchNormV3/ReadVariableOp�9batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_114/ReadVariableOp�(batch_normalization_114/ReadVariableOp_1�7batch_normalization_115/FusedBatchNormV3/ReadVariableOp�9batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_115/ReadVariableOp�(batch_normalization_115/ReadVariableOp_1�7batch_normalization_116/FusedBatchNormV3/ReadVariableOp�9batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_116/ReadVariableOp�(batch_normalization_116/ReadVariableOp_1�7batch_normalization_117/FusedBatchNormV3/ReadVariableOp�9batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_117/ReadVariableOp�(batch_normalization_117/ReadVariableOp_1�7batch_normalization_118/FusedBatchNormV3/ReadVariableOp�9batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_118/ReadVariableOp�(batch_normalization_118/ReadVariableOp_1�!conv2d_100/BiasAdd/ReadVariableOp� conv2d_100/Conv2D/ReadVariableOp�!conv2d_101/BiasAdd/ReadVariableOp� conv2d_101/Conv2D/ReadVariableOp� conv2d_96/BiasAdd/ReadVariableOp�conv2d_96/Conv2D/ReadVariableOp� conv2d_97/BiasAdd/ReadVariableOp�conv2d_97/Conv2D/ReadVariableOp� conv2d_98/BiasAdd/ReadVariableOp�conv2d_98/Conv2D/ReadVariableOp� conv2d_99/BiasAdd/ReadVariableOp�conv2d_99/Conv2D/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOp�
conv2d_96/Conv2D/ReadVariableOpReadVariableOp(conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_96/Conv2D/ReadVariableOp�
conv2d_96/Conv2DConv2Dinputs'conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_96/Conv2D�
 conv2d_96/BiasAdd/ReadVariableOpReadVariableOp)conv2d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_96/BiasAdd/ReadVariableOp�
conv2d_96/BiasAddBiasAddconv2d_96/Conv2D:output:0(conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_96/BiasAdd~
conv2d_96/ReluReluconv2d_96/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_96/Relu�
&batch_normalization_112/ReadVariableOpReadVariableOp/batch_normalization_112_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_112/ReadVariableOp�
(batch_normalization_112/ReadVariableOp_1ReadVariableOp1batch_normalization_112_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_112/ReadVariableOp_1�
7batch_normalization_112/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_112_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_112/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_112_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_112/FusedBatchNormV3FusedBatchNormV3conv2d_96/Relu:activations:0.batch_normalization_112/ReadVariableOp:value:00batch_normalization_112/ReadVariableOp_1:value:0?batch_normalization_112/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_112/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2*
(batch_normalization_112/FusedBatchNormV3�
conv2d_97/Conv2D/ReadVariableOpReadVariableOp(conv2d_97_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_97/Conv2D/ReadVariableOp�
conv2d_97/Conv2DConv2D,batch_normalization_112/FusedBatchNormV3:y:0'conv2d_97/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_97/Conv2D�
 conv2d_97/BiasAdd/ReadVariableOpReadVariableOp)conv2d_97_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_97/BiasAdd/ReadVariableOp�
conv2d_97/BiasAddBiasAddconv2d_97/Conv2D:output:0(conv2d_97/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_97/BiasAdd~
conv2d_97/ReluReluconv2d_97/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_97/Relu�
&batch_normalization_113/ReadVariableOpReadVariableOp/batch_normalization_113_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_113/ReadVariableOp�
(batch_normalization_113/ReadVariableOp_1ReadVariableOp1batch_normalization_113_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_113/ReadVariableOp_1�
7batch_normalization_113/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_113_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_113/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_113_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_113/FusedBatchNormV3FusedBatchNormV3conv2d_97/Relu:activations:0.batch_normalization_113/ReadVariableOp:value:00batch_normalization_113/ReadVariableOp_1:value:0?batch_normalization_113/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_113/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2*
(batch_normalization_113/FusedBatchNormV3�
max_pooling2d_48/MaxPoolMaxPool,batch_normalization_113/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_48/MaxPool�
conv2d_98/Conv2D/ReadVariableOpReadVariableOp(conv2d_98_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_98/Conv2D/ReadVariableOp�
conv2d_98/Conv2DConv2D!max_pooling2d_48/MaxPool:output:0'conv2d_98/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_98/Conv2D�
 conv2d_98/BiasAdd/ReadVariableOpReadVariableOp)conv2d_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_98/BiasAdd/ReadVariableOp�
conv2d_98/BiasAddBiasAddconv2d_98/Conv2D:output:0(conv2d_98/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_98/BiasAdd~
conv2d_98/ReluReluconv2d_98/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_98/Relu�
&batch_normalization_114/ReadVariableOpReadVariableOp/batch_normalization_114_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_114/ReadVariableOp�
(batch_normalization_114/ReadVariableOp_1ReadVariableOp1batch_normalization_114_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_114/ReadVariableOp_1�
7batch_normalization_114/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_114_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_114/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_114_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_114/FusedBatchNormV3FusedBatchNormV3conv2d_98/Relu:activations:0.batch_normalization_114/ReadVariableOp:value:00batch_normalization_114/ReadVariableOp_1:value:0?batch_normalization_114/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_114/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2*
(batch_normalization_114/FusedBatchNormV3�
conv2d_99/Conv2D/ReadVariableOpReadVariableOp(conv2d_99_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_99/Conv2D/ReadVariableOp�
conv2d_99/Conv2DConv2D,batch_normalization_114/FusedBatchNormV3:y:0'conv2d_99/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_99/Conv2D�
 conv2d_99/BiasAdd/ReadVariableOpReadVariableOp)conv2d_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_99/BiasAdd/ReadVariableOp�
conv2d_99/BiasAddBiasAddconv2d_99/Conv2D:output:0(conv2d_99/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_99/BiasAdd~
conv2d_99/ReluReluconv2d_99/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_99/Relu�
&batch_normalization_115/ReadVariableOpReadVariableOp/batch_normalization_115_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_115/ReadVariableOp�
(batch_normalization_115/ReadVariableOp_1ReadVariableOp1batch_normalization_115_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_115/ReadVariableOp_1�
7batch_normalization_115/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_115_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_115/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_115_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_115/FusedBatchNormV3FusedBatchNormV3conv2d_99/Relu:activations:0.batch_normalization_115/ReadVariableOp:value:00batch_normalization_115/ReadVariableOp_1:value:0?batch_normalization_115/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_115/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2*
(batch_normalization_115/FusedBatchNormV3�
max_pooling2d_49/MaxPoolMaxPool,batch_normalization_115/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_49/MaxPool�
 conv2d_100/Conv2D/ReadVariableOpReadVariableOp)conv2d_100_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02"
 conv2d_100/Conv2D/ReadVariableOp�
conv2d_100/Conv2DConv2D!max_pooling2d_49/MaxPool:output:0(conv2d_100/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_100/Conv2D�
!conv2d_100/BiasAdd/ReadVariableOpReadVariableOp*conv2d_100_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!conv2d_100/BiasAdd/ReadVariableOp�
conv2d_100/BiasAddBiasAddconv2d_100/Conv2D:output:0)conv2d_100/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_100/BiasAdd�
conv2d_100/ReluReluconv2d_100/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_100/Relu�
&batch_normalization_116/ReadVariableOpReadVariableOp/batch_normalization_116_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_116/ReadVariableOp�
(batch_normalization_116/ReadVariableOp_1ReadVariableOp1batch_normalization_116_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_116/ReadVariableOp_1�
7batch_normalization_116/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_116_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_116/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_116_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_116/FusedBatchNormV3FusedBatchNormV3conv2d_100/Relu:activations:0.batch_normalization_116/ReadVariableOp:value:00batch_normalization_116/ReadVariableOp_1:value:0?batch_normalization_116/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_116/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2*
(batch_normalization_116/FusedBatchNormV3�
 conv2d_101/Conv2D/ReadVariableOpReadVariableOp)conv2d_101_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02"
 conv2d_101/Conv2D/ReadVariableOp�
conv2d_101/Conv2DConv2D,batch_normalization_116/FusedBatchNormV3:y:0(conv2d_101/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_101/Conv2D�
!conv2d_101/BiasAdd/ReadVariableOpReadVariableOp*conv2d_101_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02#
!conv2d_101/BiasAdd/ReadVariableOp�
conv2d_101/BiasAddBiasAddconv2d_101/Conv2D:output:0)conv2d_101/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_101/BiasAdd�
conv2d_101/ReluReluconv2d_101/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_101/Relu�
&batch_normalization_117/ReadVariableOpReadVariableOp/batch_normalization_117_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_117/ReadVariableOp�
(batch_normalization_117/ReadVariableOp_1ReadVariableOp1batch_normalization_117_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_117/ReadVariableOp_1�
7batch_normalization_117/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_117_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_117/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_117_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_117/FusedBatchNormV3FusedBatchNormV3conv2d_101/Relu:activations:0.batch_normalization_117/ReadVariableOp:value:00batch_normalization_117/ReadVariableOp_1:value:0?batch_normalization_117/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_117/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2*
(batch_normalization_117/FusedBatchNormV3�
max_pooling2d_50/MaxPoolMaxPool,batch_normalization_117/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_50/MaxPool�
&batch_normalization_118/ReadVariableOpReadVariableOp/batch_normalization_118_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_118/ReadVariableOp�
(batch_normalization_118/ReadVariableOp_1ReadVariableOp1batch_normalization_118_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_118/ReadVariableOp_1�
7batch_normalization_118/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_118_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_118/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_118_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_118/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_50/MaxPool:output:0.batch_normalization_118/ReadVariableOp:value:00batch_normalization_118/ReadVariableOp_1:value:0?batch_normalization_118/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_118/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2*
(batch_normalization_118/FusedBatchNormV3u
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_16/Const�
flatten_16/ReshapeReshape,batch_normalization_118/FusedBatchNormV3:y:0flatten_16/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_16/Reshape�
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_17/MatMul/ReadVariableOp�
dense_17/MatMulMatMulflatten_16/Reshape:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_17/MatMul�
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_17/BiasAdd/ReadVariableOp�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_17/BiasAdd}
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_17/Sigmoid�
lambda_17/l2_normalize/SquareSquaredense_17/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
lambda_17/l2_normalize/Square�
,lambda_17/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,lambda_17/l2_normalize/Sum/reduction_indices�
lambda_17/l2_normalize/SumSum!lambda_17/l2_normalize/Square:y:05lambda_17/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
lambda_17/l2_normalize/Sum�
 lambda_17/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2"
 lambda_17/l2_normalize/Maximum/y�
lambda_17/l2_normalize/MaximumMaximum#lambda_17/l2_normalize/Sum:output:0)lambda_17/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2 
lambda_17/l2_normalize/Maximum�
lambda_17/l2_normalize/RsqrtRsqrt"lambda_17/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
lambda_17/l2_normalize/Rsqrt�
lambda_17/l2_normalizeMuldense_17/Sigmoid:y:0 lambda_17/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
lambda_17/l2_normalize�
IdentityIdentitylambda_17/l2_normalize:z:08^batch_normalization_112/FusedBatchNormV3/ReadVariableOp:^batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_112/ReadVariableOp)^batch_normalization_112/ReadVariableOp_18^batch_normalization_113/FusedBatchNormV3/ReadVariableOp:^batch_normalization_113/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_113/ReadVariableOp)^batch_normalization_113/ReadVariableOp_18^batch_normalization_114/FusedBatchNormV3/ReadVariableOp:^batch_normalization_114/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_114/ReadVariableOp)^batch_normalization_114/ReadVariableOp_18^batch_normalization_115/FusedBatchNormV3/ReadVariableOp:^batch_normalization_115/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_115/ReadVariableOp)^batch_normalization_115/ReadVariableOp_18^batch_normalization_116/FusedBatchNormV3/ReadVariableOp:^batch_normalization_116/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_116/ReadVariableOp)^batch_normalization_116/ReadVariableOp_18^batch_normalization_117/FusedBatchNormV3/ReadVariableOp:^batch_normalization_117/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_117/ReadVariableOp)^batch_normalization_117/ReadVariableOp_18^batch_normalization_118/FusedBatchNormV3/ReadVariableOp:^batch_normalization_118/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_118/ReadVariableOp)^batch_normalization_118/ReadVariableOp_1"^conv2d_100/BiasAdd/ReadVariableOp!^conv2d_100/Conv2D/ReadVariableOp"^conv2d_101/BiasAdd/ReadVariableOp!^conv2d_101/Conv2D/ReadVariableOp!^conv2d_96/BiasAdd/ReadVariableOp ^conv2d_96/Conv2D/ReadVariableOp!^conv2d_97/BiasAdd/ReadVariableOp ^conv2d_97/Conv2D/ReadVariableOp!^conv2d_98/BiasAdd/ReadVariableOp ^conv2d_98/Conv2D/ReadVariableOp!^conv2d_99/BiasAdd/ReadVariableOp ^conv2d_99/Conv2D/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2r
7batch_normalization_112/FusedBatchNormV3/ReadVariableOp7batch_normalization_112/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_112/FusedBatchNormV3/ReadVariableOp_19batch_normalization_112/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_112/ReadVariableOp&batch_normalization_112/ReadVariableOp2T
(batch_normalization_112/ReadVariableOp_1(batch_normalization_112/ReadVariableOp_12r
7batch_normalization_113/FusedBatchNormV3/ReadVariableOp7batch_normalization_113/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_113/FusedBatchNormV3/ReadVariableOp_19batch_normalization_113/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_113/ReadVariableOp&batch_normalization_113/ReadVariableOp2T
(batch_normalization_113/ReadVariableOp_1(batch_normalization_113/ReadVariableOp_12r
7batch_normalization_114/FusedBatchNormV3/ReadVariableOp7batch_normalization_114/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_114/FusedBatchNormV3/ReadVariableOp_19batch_normalization_114/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_114/ReadVariableOp&batch_normalization_114/ReadVariableOp2T
(batch_normalization_114/ReadVariableOp_1(batch_normalization_114/ReadVariableOp_12r
7batch_normalization_115/FusedBatchNormV3/ReadVariableOp7batch_normalization_115/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_115/FusedBatchNormV3/ReadVariableOp_19batch_normalization_115/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_115/ReadVariableOp&batch_normalization_115/ReadVariableOp2T
(batch_normalization_115/ReadVariableOp_1(batch_normalization_115/ReadVariableOp_12r
7batch_normalization_116/FusedBatchNormV3/ReadVariableOp7batch_normalization_116/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_116/FusedBatchNormV3/ReadVariableOp_19batch_normalization_116/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_116/ReadVariableOp&batch_normalization_116/ReadVariableOp2T
(batch_normalization_116/ReadVariableOp_1(batch_normalization_116/ReadVariableOp_12r
7batch_normalization_117/FusedBatchNormV3/ReadVariableOp7batch_normalization_117/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_117/FusedBatchNormV3/ReadVariableOp_19batch_normalization_117/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_117/ReadVariableOp&batch_normalization_117/ReadVariableOp2T
(batch_normalization_117/ReadVariableOp_1(batch_normalization_117/ReadVariableOp_12r
7batch_normalization_118/FusedBatchNormV3/ReadVariableOp7batch_normalization_118/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_118/FusedBatchNormV3/ReadVariableOp_19batch_normalization_118/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_118/ReadVariableOp&batch_normalization_118/ReadVariableOp2T
(batch_normalization_118/ReadVariableOp_1(batch_normalization_118/ReadVariableOp_12F
!conv2d_100/BiasAdd/ReadVariableOp!conv2d_100/BiasAdd/ReadVariableOp2D
 conv2d_100/Conv2D/ReadVariableOp conv2d_100/Conv2D/ReadVariableOp2F
!conv2d_101/BiasAdd/ReadVariableOp!conv2d_101/BiasAdd/ReadVariableOp2D
 conv2d_101/Conv2D/ReadVariableOp conv2d_101/Conv2D/ReadVariableOp2D
 conv2d_96/BiasAdd/ReadVariableOp conv2d_96/BiasAdd/ReadVariableOp2B
conv2d_96/Conv2D/ReadVariableOpconv2d_96/Conv2D/ReadVariableOp2D
 conv2d_97/BiasAdd/ReadVariableOp conv2d_97/BiasAdd/ReadVariableOp2B
conv2d_97/Conv2D/ReadVariableOpconv2d_97/Conv2D/ReadVariableOp2D
 conv2d_98/BiasAdd/ReadVariableOp conv2d_98/BiasAdd/ReadVariableOp2B
conv2d_98/Conv2D/ReadVariableOpconv2d_98/Conv2D/ReadVariableOp2D
 conv2d_99/BiasAdd/ReadVariableOp conv2d_99/BiasAdd/ReadVariableOp2B
conv2d_99/Conv2D/ReadVariableOpconv2d_99/Conv2D/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_48_layer_call_fn_226773

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2267672
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_112_layer_call_fn_229318

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_2266152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_227787

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_226970

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_228665
conv2d_96_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_96_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_2265532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_96_input
�
�
.__inference_sequential_16_layer_call_fn_229094

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*>
_read_only_resource_inputs 
	
 !"%&)**-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_2282912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_113_layer_call_fn_229402

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_2267192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_227668

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229897

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_sequential_16_layer_call_fn_228378
conv2d_96_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_96_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*>
_read_only_resource_inputs 
	
 !"%&)**-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_2282912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_96_input
�

a
E__inference_lambda_17_layer_call_and_return_conditional_losses_228049

inputs
identityo
l2_normalize/SquareSquareinputs*
T0*(
_output_shapes
:����������2
l2_normalize/Square�
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2$
"l2_normalize/Sum/reduction_indices�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
l2_normalize/Sumu
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2
l2_normalize/Maximum/y�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2
l2_normalize/Maximum}
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
l2_normalize/Rsqrtv
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
l2_normalizee
IdentityIdentityl2_normalize:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_226987

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229453

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_227943

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_227467

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
b
F__inference_flatten_16_layer_call_and_return_conditional_losses_230205

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�k
�
I__inference_sequential_16_layer_call_and_return_conditional_losses_228181
conv2d_96_input
conv2d_96_228077
conv2d_96_228079"
batch_normalization_112_228082"
batch_normalization_112_228084"
batch_normalization_112_228086"
batch_normalization_112_228088
conv2d_97_228091
conv2d_97_228093"
batch_normalization_113_228096"
batch_normalization_113_228098"
batch_normalization_113_228100"
batch_normalization_113_228102
conv2d_98_228106
conv2d_98_228108"
batch_normalization_114_228111"
batch_normalization_114_228113"
batch_normalization_114_228115"
batch_normalization_114_228117
conv2d_99_228120
conv2d_99_228122"
batch_normalization_115_228125"
batch_normalization_115_228127"
batch_normalization_115_228129"
batch_normalization_115_228131
conv2d_100_228135
conv2d_100_228137"
batch_normalization_116_228140"
batch_normalization_116_228142"
batch_normalization_116_228144"
batch_normalization_116_228146
conv2d_101_228149
conv2d_101_228151"
batch_normalization_117_228154"
batch_normalization_117_228156"
batch_normalization_117_228158"
batch_normalization_117_228160"
batch_normalization_118_228164"
batch_normalization_118_228166"
batch_normalization_118_228168"
batch_normalization_118_228170
dense_17_228174
dense_17_228176
identity��/batch_normalization_112/StatefulPartitionedCall�/batch_normalization_113/StatefulPartitionedCall�/batch_normalization_114/StatefulPartitionedCall�/batch_normalization_115/StatefulPartitionedCall�/batch_normalization_116/StatefulPartitionedCall�/batch_normalization_117/StatefulPartitionedCall�/batch_normalization_118/StatefulPartitionedCall�"conv2d_100/StatefulPartitionedCall�"conv2d_101/StatefulPartitionedCall�!conv2d_96/StatefulPartitionedCall�!conv2d_97/StatefulPartitionedCall�!conv2d_98/StatefulPartitionedCall�!conv2d_99/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCallconv2d_96_inputconv2d_96_228077conv2d_96_228079*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_2273322#
!conv2d_96/StatefulPartitionedCall�
/batch_normalization_112/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0batch_normalization_112_228082batch_normalization_112_228084batch_normalization_112_228086batch_normalization_112_228088*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_22738521
/batch_normalization_112/StatefulPartitionedCall�
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_112/StatefulPartitionedCall:output:0conv2d_97_228091conv2d_97_228093*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_2274322#
!conv2d_97/StatefulPartitionedCall�
/batch_normalization_113/StatefulPartitionedCallStatefulPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0batch_normalization_113_228096batch_normalization_113_228098batch_normalization_113_228100batch_normalization_113_228102*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_22748521
/batch_normalization_113/StatefulPartitionedCall�
 max_pooling2d_48/PartitionedCallPartitionedCall8batch_normalization_113/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_2267672"
 max_pooling2d_48/PartitionedCall�
!conv2d_98/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0conv2d_98_228106conv2d_98_228108*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_98_layer_call_and_return_conditional_losses_2275332#
!conv2d_98/StatefulPartitionedCall�
/batch_normalization_114/StatefulPartitionedCallStatefulPartitionedCall*conv2d_98/StatefulPartitionedCall:output:0batch_normalization_114_228111batch_normalization_114_228113batch_normalization_114_228115batch_normalization_114_228117*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_22758621
/batch_normalization_114/StatefulPartitionedCall�
!conv2d_99/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_114/StatefulPartitionedCall:output:0conv2d_99_228120conv2d_99_228122*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_99_layer_call_and_return_conditional_losses_2276332#
!conv2d_99/StatefulPartitionedCall�
/batch_normalization_115/StatefulPartitionedCallStatefulPartitionedCall*conv2d_99/StatefulPartitionedCall:output:0batch_normalization_115_228125batch_normalization_115_228127batch_normalization_115_228129batch_normalization_115_228131*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_22768621
/batch_normalization_115/StatefulPartitionedCall�
 max_pooling2d_49/PartitionedCallPartitionedCall8batch_normalization_115/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_2269872"
 max_pooling2d_49/PartitionedCall�
"conv2d_100/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0conv2d_100_228135conv2d_100_228137*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_100_layer_call_and_return_conditional_losses_2277342$
"conv2d_100/StatefulPartitionedCall�
/batch_normalization_116/StatefulPartitionedCallStatefulPartitionedCall+conv2d_100/StatefulPartitionedCall:output:0batch_normalization_116_228140batch_normalization_116_228142batch_normalization_116_228144batch_normalization_116_228146*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_22778721
/batch_normalization_116/StatefulPartitionedCall�
"conv2d_101/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_116/StatefulPartitionedCall:output:0conv2d_101_228149conv2d_101_228151*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_101_layer_call_and_return_conditional_losses_2278342$
"conv2d_101/StatefulPartitionedCall�
/batch_normalization_117/StatefulPartitionedCallStatefulPartitionedCall+conv2d_101/StatefulPartitionedCall:output:0batch_normalization_117_228154batch_normalization_117_228156batch_normalization_117_228158batch_normalization_117_228160*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_22788721
/batch_normalization_117/StatefulPartitionedCall�
 max_pooling2d_50/PartitionedCallPartitionedCall8batch_normalization_117/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2272072"
 max_pooling2d_50/PartitionedCall�
/batch_normalization_118/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0batch_normalization_118_228164batch_normalization_118_228166batch_normalization_118_228168batch_normalization_118_228170*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_22796121
/batch_normalization_118/StatefulPartitionedCall�
flatten_16/PartitionedCallPartitionedCall8batch_normalization_118/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_16_layer_call_and_return_conditional_losses_2280032
flatten_16/PartitionedCall�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_17_228174dense_17_228176*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_2280222"
 dense_17/StatefulPartitionedCall�
lambda_17/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lambda_17_layer_call_and_return_conditional_losses_2280602
lambda_17/PartitionedCall�
IdentityIdentity"lambda_17/PartitionedCall:output:00^batch_normalization_112/StatefulPartitionedCall0^batch_normalization_113/StatefulPartitionedCall0^batch_normalization_114/StatefulPartitionedCall0^batch_normalization_115/StatefulPartitionedCall0^batch_normalization_116/StatefulPartitionedCall0^batch_normalization_117/StatefulPartitionedCall0^batch_normalization_118/StatefulPartitionedCall#^conv2d_100/StatefulPartitionedCall#^conv2d_101/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall"^conv2d_98/StatefulPartitionedCall"^conv2d_99/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2b
/batch_normalization_112/StatefulPartitionedCall/batch_normalization_112/StatefulPartitionedCall2b
/batch_normalization_113/StatefulPartitionedCall/batch_normalization_113/StatefulPartitionedCall2b
/batch_normalization_114/StatefulPartitionedCall/batch_normalization_114/StatefulPartitionedCall2b
/batch_normalization_115/StatefulPartitionedCall/batch_normalization_115/StatefulPartitionedCall2b
/batch_normalization_116/StatefulPartitionedCall/batch_normalization_116/StatefulPartitionedCall2b
/batch_normalization_117/StatefulPartitionedCall/batch_normalization_117/StatefulPartitionedCall2b
/batch_normalization_118/StatefulPartitionedCall/batch_normalization_118/StatefulPartitionedCall2H
"conv2d_100/StatefulPartitionedCall"conv2d_100/StatefulPartitionedCall2H
"conv2d_101/StatefulPartitionedCall"conv2d_101/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2F
!conv2d_98/StatefulPartitionedCall!conv2d_98/StatefulPartitionedCall2F
!conv2d_99/StatefulPartitionedCall!conv2d_99/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_96_input
�
�
+__inference_conv2d_101_layer_call_fn_229943

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_101_layer_call_and_return_conditional_losses_2278342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_116_layer_call_fn_229859

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_2270862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_118_layer_call_fn_230199

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_2279612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_50_layer_call_fn_227213

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_2272072
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_112_layer_call_fn_229254

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_2273672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229731

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_227086

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_117_layer_call_fn_230071

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_2278872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
S
conv2d_96_input@
!serving_default_conv2d_96_input:0���������  >
	lambda_171
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
��
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer-14
layer_with_weights-12
layer-15
layer-16
layer_with_weights-13
layer-17
layer-18
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"��
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_96_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_96", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_112", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_97", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_113", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_48", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_98", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_114", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_99", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_115", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_100", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_116", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_101", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_117", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_118", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Lambda", "config": {"name": "lambda_17", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPof\nPGlweXRob24taW5wdXQtODEtZDgwNjgwM2NmNWE2PtoIPGxhbWJkYT4YAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_96_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_96", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_112", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_97", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_113", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_48", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_98", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_114", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_99", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_115", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_100", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_116", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_101", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_117", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_118", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Lambda", "config": {"name": "lambda_17", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPof\nPGlweXRob24taW5wdXQtODEtZDgwNjgwM2NmNWE2PtoIPGxhbWJkYT4YAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}}}
�


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_96", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_96", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
�	
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%trainable_variables
&	variables
'	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_112", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_112", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�	

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_97", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_97", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�	
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3regularization_losses
4trainable_variables
5	variables
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_113", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_113", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_48", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_98", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_98", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
�	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_114", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_114", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�	

Jkernel
Kbias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_99", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_99", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�	
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_115", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_115", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

]kernel
^bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_100", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_100", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
�	
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance
hregularization_losses
itrainable_variables
j	variables
k	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_116", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_116", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�	

lkernel
mbias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_101", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_101", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�	
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_117", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_117", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�
{regularization_losses
|trainable_variables
}	variables
~	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	
axis

�gamma
	�beta
�moving_mean
�moving_variance
�regularization_losses
�trainable_variables
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_118", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_118", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�kernel
	�bias
�regularization_losses
�trainable_variables
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "lambda_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_17", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPof\nPGlweXRob24taW5wdXQtODEtZDgwNjgwM2NmNWE2PtoIPGxhbWJkYT4YAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
 "
trackable_list_wrapper
�
0
1
 2
!3
(4
)5
/6
07
;8
<9
B10
C11
J12
K13
Q14
R15
]16
^17
d18
e19
l20
m21
s22
t23
�24
�25
�26
�27"
trackable_list_wrapper
�
0
1
 2
!3
"4
#5
(6
)7
/8
09
110
211
;12
<13
B14
C15
D16
E17
J18
K19
Q20
R21
S22
T23
]24
^25
d26
e27
f28
g29
l30
m31
s32
t33
u34
v35
�36
�37
�38
�39
�40
�41"
trackable_list_wrapper
�
�layer_metrics
regularization_losses
trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
	variables
�layers
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
*:( 2conv2d_96/kernel
: 2conv2d_96/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
�layer_metrics
regularization_losses
trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:) 2batch_normalization_112/gamma
*:( 2batch_normalization_112/beta
3:1  (2#batch_normalization_112/moving_mean
7:5  (2'batch_normalization_112/moving_variance
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
<
 0
!1
"2
#3"
trackable_list_wrapper
�
�layer_metrics
$regularization_losses
%trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
&	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_97/kernel
: 2conv2d_97/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
�layer_metrics
*regularization_losses
+trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
,	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:) 2batch_normalization_113/gamma
*:( 2batch_normalization_113/beta
3:1  (2#batch_normalization_113/moving_mean
7:5  (2'batch_normalization_113/moving_variance
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
�
�layer_metrics
3regularization_losses
4trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
5	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
7regularization_losses
8trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
9	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_98/kernel
:@2conv2d_98/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
�
�layer_metrics
=regularization_losses
>trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
?	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)@2batch_normalization_114/gamma
*:(@2batch_normalization_114/beta
3:1@ (2#batch_normalization_114/moving_mean
7:5@ (2'batch_normalization_114/moving_variance
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
�
�layer_metrics
Fregularization_losses
Gtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
H	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_99/kernel
:@2conv2d_99/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
�
�layer_metrics
Lregularization_losses
Mtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
N	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)@2batch_normalization_115/gamma
*:(@2batch_normalization_115/beta
3:1@ (2#batch_normalization_115/moving_mean
7:5@ (2'batch_normalization_115/moving_variance
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
<
Q0
R1
S2
T3"
trackable_list_wrapper
�
�layer_metrics
Uregularization_losses
Vtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
W	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
Yregularization_losses
Ztrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
[	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*@�2conv2d_100/kernel
:�2conv2d_100/bias
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
�
�layer_metrics
_regularization_losses
`trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
a	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*�2batch_normalization_116/gamma
+:)�2batch_normalization_116/beta
4:2� (2#batch_normalization_116/moving_mean
8:6� (2'batch_normalization_116/moving_variance
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
<
d0
e1
f2
g3"
trackable_list_wrapper
�
�layer_metrics
hregularization_losses
itrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
j	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+��2conv2d_101/kernel
:�2conv2d_101/bias
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
�
�layer_metrics
nregularization_losses
otrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
p	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*�2batch_normalization_117/gamma
+:)�2batch_normalization_117/beta
4:2� (2#batch_normalization_117/moving_mean
8:6� (2'batch_normalization_117/moving_variance
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
<
s0
t1
u2
v3"
trackable_list_wrapper
�
�layer_metrics
wregularization_losses
xtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
y	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
{regularization_losses
|trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
}	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*�2batch_normalization_118/gamma
+:)�2batch_normalization_118/beta
4:2� (2#batch_normalization_118/moving_mean
8:6� (2'batch_normalization_118/moving_variance
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�layer_metrics
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
�	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
�	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_17/kernel
:�2dense_17/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�layer_metrics
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
�	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�metrics
�	variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
"0
#1
12
23
D4
E5
S6
T7
f8
g9
u10
v11
�12
�13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
I__inference_sequential_16_layer_call_and_return_conditional_losses_228842
I__inference_sequential_16_layer_call_and_return_conditional_losses_228181
I__inference_sequential_16_layer_call_and_return_conditional_losses_228074
I__inference_sequential_16_layer_call_and_return_conditional_losses_229005�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_226553�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *6�3
1�.
conv2d_96_input���������  
�2�
.__inference_sequential_16_layer_call_fn_228574
.__inference_sequential_16_layer_call_fn_228378
.__inference_sequential_16_layer_call_fn_229183
.__inference_sequential_16_layer_call_fn_229094�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_96_layer_call_and_return_conditional_losses_229194�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_96_layer_call_fn_229203�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229241
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229223
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229305
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229287�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_112_layer_call_fn_229267
8__inference_batch_normalization_112_layer_call_fn_229331
8__inference_batch_normalization_112_layer_call_fn_229318
8__inference_batch_normalization_112_layer_call_fn_229254�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_97_layer_call_and_return_conditional_losses_229342�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_97_layer_call_fn_229351�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229453
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229389
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229435
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229371�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_113_layer_call_fn_229466
8__inference_batch_normalization_113_layer_call_fn_229415
8__inference_batch_normalization_113_layer_call_fn_229402
8__inference_batch_normalization_113_layer_call_fn_229479�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_226767�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
1__inference_max_pooling2d_48_layer_call_fn_226773�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
E__inference_conv2d_98_layer_call_and_return_conditional_losses_229490�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_98_layer_call_fn_229499�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229537
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229519
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229601
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229583�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_114_layer_call_fn_229627
8__inference_batch_normalization_114_layer_call_fn_229563
8__inference_batch_normalization_114_layer_call_fn_229550
8__inference_batch_normalization_114_layer_call_fn_229614�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_conv2d_99_layer_call_and_return_conditional_losses_229638�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_99_layer_call_fn_229647�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229667
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229749
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229731
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229685�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_115_layer_call_fn_229762
8__inference_batch_normalization_115_layer_call_fn_229775
8__inference_batch_normalization_115_layer_call_fn_229698
8__inference_batch_normalization_115_layer_call_fn_229711�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_226987�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
1__inference_max_pooling2d_49_layer_call_fn_226993�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
F__inference_conv2d_100_layer_call_and_return_conditional_losses_229786�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv2d_100_layer_call_fn_229795�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229815
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229879
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229897
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229833�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_116_layer_call_fn_229846
8__inference_batch_normalization_116_layer_call_fn_229910
8__inference_batch_normalization_116_layer_call_fn_229923
8__inference_batch_normalization_116_layer_call_fn_229859�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_conv2d_101_layer_call_and_return_conditional_losses_229934�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv2d_101_layer_call_fn_229943�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_230045
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_230027
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_229963
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_229981�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_117_layer_call_fn_229994
8__inference_batch_normalization_117_layer_call_fn_230058
8__inference_batch_normalization_117_layer_call_fn_230071
8__inference_batch_normalization_117_layer_call_fn_230007�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_227207�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
1__inference_max_pooling2d_50_layer_call_fn_227213�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230173
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230091
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230109
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230155�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
8__inference_batch_normalization_118_layer_call_fn_230135
8__inference_batch_normalization_118_layer_call_fn_230122
8__inference_batch_normalization_118_layer_call_fn_230186
8__inference_batch_normalization_118_layer_call_fn_230199�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_flatten_16_layer_call_and_return_conditional_losses_230205�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_flatten_16_layer_call_fn_230210�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_17_layer_call_and_return_conditional_losses_230221�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_17_layer_call_fn_230230�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_lambda_17_layer_call_and_return_conditional_losses_230252
E__inference_lambda_17_layer_call_and_return_conditional_losses_230241�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_lambda_17_layer_call_fn_230262
*__inference_lambda_17_layer_call_fn_230257�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
$__inference_signature_wrapper_228665conv2d_96_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_226553�0 !"#()/012;<BCDEJKQRST]^defglmstuv������@�=
6�3
1�.
conv2d_96_input���������  
� "6�3
1
	lambda_17$�!
	lambda_17�����������
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229223r !"#;�8
1�.
(�%
inputs���������   
p
� "-�*
#� 
0���������   
� �
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229241r !"#;�8
1�.
(�%
inputs���������   
p 
� "-�*
#� 
0���������   
� �
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229287� !"#M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_229305� !"#M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
8__inference_batch_normalization_112_layer_call_fn_229254e !"#;�8
1�.
(�%
inputs���������   
p
� " ����������   �
8__inference_batch_normalization_112_layer_call_fn_229267e !"#;�8
1�.
(�%
inputs���������   
p 
� " ����������   �
8__inference_batch_normalization_112_layer_call_fn_229318� !"#M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
8__inference_batch_normalization_112_layer_call_fn_229331� !"#M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229371�/012M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229389�/012M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229435r/012;�8
1�.
(�%
inputs���������   
p
� "-�*
#� 
0���������   
� �
S__inference_batch_normalization_113_layer_call_and_return_conditional_losses_229453r/012;�8
1�.
(�%
inputs���������   
p 
� "-�*
#� 
0���������   
� �
8__inference_batch_normalization_113_layer_call_fn_229402�/012M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
8__inference_batch_normalization_113_layer_call_fn_229415�/012M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
8__inference_batch_normalization_113_layer_call_fn_229466e/012;�8
1�.
(�%
inputs���������   
p
� " ����������   �
8__inference_batch_normalization_113_layer_call_fn_229479e/012;�8
1�.
(�%
inputs���������   
p 
� " ����������   �
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229519�BCDEM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229537�BCDEM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229583rBCDE;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
S__inference_batch_normalization_114_layer_call_and_return_conditional_losses_229601rBCDE;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
8__inference_batch_normalization_114_layer_call_fn_229550�BCDEM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
8__inference_batch_normalization_114_layer_call_fn_229563�BCDEM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
8__inference_batch_normalization_114_layer_call_fn_229614eBCDE;�8
1�.
(�%
inputs���������@
p
� " ����������@�
8__inference_batch_normalization_114_layer_call_fn_229627eBCDE;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229667�QRSTM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229685�QRSTM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229731rQRST;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
S__inference_batch_normalization_115_layer_call_and_return_conditional_losses_229749rQRST;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
8__inference_batch_normalization_115_layer_call_fn_229698�QRSTM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
8__inference_batch_normalization_115_layer_call_fn_229711�QRSTM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
8__inference_batch_normalization_115_layer_call_fn_229762eQRST;�8
1�.
(�%
inputs���������@
p
� " ����������@�
8__inference_batch_normalization_115_layer_call_fn_229775eQRST;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229815�defgN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229833�defgN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229879tdefg<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
S__inference_batch_normalization_116_layer_call_and_return_conditional_losses_229897tdefg<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
8__inference_batch_normalization_116_layer_call_fn_229846�defgN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_116_layer_call_fn_229859�defgN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
8__inference_batch_normalization_116_layer_call_fn_229910gdefg<�9
2�/
)�&
inputs����������
p
� "!������������
8__inference_batch_normalization_116_layer_call_fn_229923gdefg<�9
2�/
)�&
inputs����������
p 
� "!������������
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_229963�stuvN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_229981�stuvN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_230027tstuv<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
S__inference_batch_normalization_117_layer_call_and_return_conditional_losses_230045tstuv<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
8__inference_batch_normalization_117_layer_call_fn_229994�stuvN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_117_layer_call_fn_230007�stuvN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
8__inference_batch_normalization_117_layer_call_fn_230058gstuv<�9
2�/
)�&
inputs����������
p
� "!������������
8__inference_batch_normalization_117_layer_call_fn_230071gstuv<�9
2�/
)�&
inputs����������
p 
� "!������������
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230091�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230109�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230155x����<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
S__inference_batch_normalization_118_layer_call_and_return_conditional_losses_230173x����<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
8__inference_batch_normalization_118_layer_call_fn_230122�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_118_layer_call_fn_230135�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
8__inference_batch_normalization_118_layer_call_fn_230186k����<�9
2�/
)�&
inputs����������
p
� "!������������
8__inference_batch_normalization_118_layer_call_fn_230199k����<�9
2�/
)�&
inputs����������
p 
� "!������������
F__inference_conv2d_100_layer_call_and_return_conditional_losses_229786m]^7�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
+__inference_conv2d_100_layer_call_fn_229795`]^7�4
-�*
(�%
inputs���������@
� "!������������
F__inference_conv2d_101_layer_call_and_return_conditional_losses_229934nlm8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
+__inference_conv2d_101_layer_call_fn_229943alm8�5
.�+
)�&
inputs����������
� "!������������
E__inference_conv2d_96_layer_call_and_return_conditional_losses_229194l7�4
-�*
(�%
inputs���������  
� "-�*
#� 
0���������   
� �
*__inference_conv2d_96_layer_call_fn_229203_7�4
-�*
(�%
inputs���������  
� " ����������   �
E__inference_conv2d_97_layer_call_and_return_conditional_losses_229342l()7�4
-�*
(�%
inputs���������   
� "-�*
#� 
0���������   
� �
*__inference_conv2d_97_layer_call_fn_229351_()7�4
-�*
(�%
inputs���������   
� " ����������   �
E__inference_conv2d_98_layer_call_and_return_conditional_losses_229490l;<7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
*__inference_conv2d_98_layer_call_fn_229499_;<7�4
-�*
(�%
inputs��������� 
� " ����������@�
E__inference_conv2d_99_layer_call_and_return_conditional_losses_229638lJK7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
*__inference_conv2d_99_layer_call_fn_229647_JK7�4
-�*
(�%
inputs���������@
� " ����������@�
D__inference_dense_17_layer_call_and_return_conditional_losses_230221`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
)__inference_dense_17_layer_call_fn_230230S��0�-
&�#
!�
inputs����������
� "������������
F__inference_flatten_16_layer_call_and_return_conditional_losses_230205b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
+__inference_flatten_16_layer_call_fn_230210U8�5
.�+
)�&
inputs����������
� "������������
E__inference_lambda_17_layer_call_and_return_conditional_losses_230241b8�5
.�+
!�
inputs����������

 
p
� "&�#
�
0����������
� �
E__inference_lambda_17_layer_call_and_return_conditional_losses_230252b8�5
.�+
!�
inputs����������

 
p 
� "&�#
�
0����������
� �
*__inference_lambda_17_layer_call_fn_230257U8�5
.�+
!�
inputs����������

 
p
� "������������
*__inference_lambda_17_layer_call_fn_230262U8�5
.�+
!�
inputs����������

 
p 
� "������������
L__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_226767�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_48_layer_call_fn_226773�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_226987�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_49_layer_call_fn_226993�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_227207�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_50_layer_call_fn_227213�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_16_layer_call_and_return_conditional_losses_228074�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_96_input���������  
p

 
� "&�#
�
0����������
� �
I__inference_sequential_16_layer_call_and_return_conditional_losses_228181�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_96_input���������  
p 

 
� "&�#
�
0����������
� �
I__inference_sequential_16_layer_call_and_return_conditional_losses_228842�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
5�2
(�%
inputs���������  
p

 
� "&�#
�
0����������
� �
I__inference_sequential_16_layer_call_and_return_conditional_losses_229005�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
5�2
(�%
inputs���������  
p 

 
� "&�#
�
0����������
� �
.__inference_sequential_16_layer_call_fn_228378�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_96_input���������  
p

 
� "������������
.__inference_sequential_16_layer_call_fn_228574�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_96_input���������  
p 

 
� "������������
.__inference_sequential_16_layer_call_fn_229094�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
5�2
(�%
inputs���������  
p

 
� "������������
.__inference_sequential_16_layer_call_fn_229183�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
5�2
(�%
inputs���������  
p 

 
� "������������
$__inference_signature_wrapper_228665�0 !"#()/012;<BCDEJKQRST]^defglmstuv������S�P
� 
I�F
D
conv2d_96_input1�.
conv2d_96_input���������  "6�3
1
	lambda_17$�!
	lambda_17����������