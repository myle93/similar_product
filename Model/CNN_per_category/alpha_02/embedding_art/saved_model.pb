�� 
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
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
�
conv2d_90/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_90/kernel
}
$conv2d_90/kernel/Read/ReadVariableOpReadVariableOpconv2d_90/kernel*&
_output_shapes
: *
dtype0
t
conv2d_90/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_90/bias
m
"conv2d_90/bias/Read/ReadVariableOpReadVariableOpconv2d_90/bias*
_output_shapes
: *
dtype0
�
batch_normalization_105/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_105/gamma
�
1batch_normalization_105/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_105/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_105/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_105/beta
�
0batch_normalization_105/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_105/beta*
_output_shapes
: *
dtype0
�
#batch_normalization_105/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_105/moving_mean
�
7batch_normalization_105/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_105/moving_mean*
_output_shapes
: *
dtype0
�
'batch_normalization_105/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_105/moving_variance
�
;batch_normalization_105/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_105/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_91/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_91/kernel
}
$conv2d_91/kernel/Read/ReadVariableOpReadVariableOpconv2d_91/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_91/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_91/bias
m
"conv2d_91/bias/Read/ReadVariableOpReadVariableOpconv2d_91/bias*
_output_shapes
: *
dtype0
�
batch_normalization_106/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_106/gamma
�
1batch_normalization_106/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_106/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_106/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_106/beta
�
0batch_normalization_106/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_106/beta*
_output_shapes
: *
dtype0
�
#batch_normalization_106/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization_106/moving_mean
�
7batch_normalization_106/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_106/moving_mean*
_output_shapes
: *
dtype0
�
'batch_normalization_106/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'batch_normalization_106/moving_variance
�
;batch_normalization_106/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_106/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_92/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_92/kernel
}
$conv2d_92/kernel/Read/ReadVariableOpReadVariableOpconv2d_92/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_92/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_92/bias
m
"conv2d_92/bias/Read/ReadVariableOpReadVariableOpconv2d_92/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_107/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_107/gamma
�
1batch_normalization_107/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_107/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_107/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_107/beta
�
0batch_normalization_107/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_107/beta*
_output_shapes
:@*
dtype0
�
#batch_normalization_107/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_107/moving_mean
�
7batch_normalization_107/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_107/moving_mean*
_output_shapes
:@*
dtype0
�
'batch_normalization_107/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_107/moving_variance
�
;batch_normalization_107/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_107/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_93/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_93/kernel
}
$conv2d_93/kernel/Read/ReadVariableOpReadVariableOpconv2d_93/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_93/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_93/bias
m
"conv2d_93/bias/Read/ReadVariableOpReadVariableOpconv2d_93/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_108/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_108/gamma
�
1batch_normalization_108/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_108/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_108/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_108/beta
�
0batch_normalization_108/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_108/beta*
_output_shapes
:@*
dtype0
�
#batch_normalization_108/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization_108/moving_mean
�
7batch_normalization_108/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_108/moving_mean*
_output_shapes
:@*
dtype0
�
'batch_normalization_108/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'batch_normalization_108/moving_variance
�
;batch_normalization_108/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_108/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_94/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameconv2d_94/kernel
~
$conv2d_94/kernel/Read/ReadVariableOpReadVariableOpconv2d_94/kernel*'
_output_shapes
:@�*
dtype0
u
conv2d_94/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_94/bias
n
"conv2d_94/bias/Read/ReadVariableOpReadVariableOpconv2d_94/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_109/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namebatch_normalization_109/gamma
�
1batch_normalization_109/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_109/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_109/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_109/beta
�
0batch_normalization_109/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_109/beta*
_output_shapes	
:�*
dtype0
�
#batch_normalization_109/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_109/moving_mean
�
7batch_normalization_109/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_109/moving_mean*
_output_shapes	
:�*
dtype0
�
'batch_normalization_109/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'batch_normalization_109/moving_variance
�
;batch_normalization_109/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_109/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_95/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_95/kernel

$conv2d_95/kernel/Read/ReadVariableOpReadVariableOpconv2d_95/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_95/bias
n
"conv2d_95/bias/Read/ReadVariableOpReadVariableOpconv2d_95/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_110/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namebatch_normalization_110/gamma
�
1batch_normalization_110/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_110/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_110/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_110/beta
�
0batch_normalization_110/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_110/beta*
_output_shapes	
:�*
dtype0
�
#batch_normalization_110/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_110/moving_mean
�
7batch_normalization_110/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_110/moving_mean*
_output_shapes	
:�*
dtype0
�
'batch_normalization_110/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'batch_normalization_110/moving_variance
�
;batch_normalization_110/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_110/moving_variance*
_output_shapes	
:�*
dtype0
�
batch_normalization_111/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namebatch_normalization_111/gamma
�
1batch_normalization_111/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_111/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_111/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_111/beta
�
0batch_normalization_111/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_111/beta*
_output_shapes	
:�*
dtype0
�
#batch_normalization_111/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization_111/moving_mean
�
7batch_normalization_111/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_111/moving_mean*
_output_shapes	
:�*
dtype0
�
'batch_normalization_111/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'batch_normalization_111/moving_variance
�
;batch_normalization_111/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_111/moving_variance*
_output_shapes	
:�*
dtype0
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
��*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
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
VARIABLE_VALUEconv2d_90/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_90/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_105/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_105/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_105/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_105/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_91/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_91/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_106/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_106/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_106/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_106/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_92/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_92/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_107/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_107/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_107/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_107/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_93/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_93/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_108/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_108/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_108/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_108/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEconv2d_94/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_94/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_109/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_109/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_109/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_109/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
][
VARIABLE_VALUEconv2d_95/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_95/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_110/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_110/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#batch_normalization_110/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'batch_normalization_110/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_111/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_111/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#batch_normalization_111/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'batch_normalization_111/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_16/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_16/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_conv2d_90_inputPlaceholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_90_inputconv2d_90/kernelconv2d_90/biasbatch_normalization_105/gammabatch_normalization_105/beta#batch_normalization_105/moving_mean'batch_normalization_105/moving_varianceconv2d_91/kernelconv2d_91/biasbatch_normalization_106/gammabatch_normalization_106/beta#batch_normalization_106/moving_mean'batch_normalization_106/moving_varianceconv2d_92/kernelconv2d_92/biasbatch_normalization_107/gammabatch_normalization_107/beta#batch_normalization_107/moving_mean'batch_normalization_107/moving_varianceconv2d_93/kernelconv2d_93/biasbatch_normalization_108/gammabatch_normalization_108/beta#batch_normalization_108/moving_mean'batch_normalization_108/moving_varianceconv2d_94/kernelconv2d_94/biasbatch_normalization_109/gammabatch_normalization_109/beta#batch_normalization_109/moving_mean'batch_normalization_109/moving_varianceconv2d_95/kernelconv2d_95/biasbatch_normalization_110/gammabatch_normalization_110/beta#batch_normalization_110/moving_mean'batch_normalization_110/moving_variancebatch_normalization_111/gammabatch_normalization_111/beta#batch_normalization_111/moving_mean'batch_normalization_111/moving_variancedense_16/kerneldense_16/bias*6
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
$__inference_signature_wrapper_219342
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_90/kernel/Read/ReadVariableOp"conv2d_90/bias/Read/ReadVariableOp1batch_normalization_105/gamma/Read/ReadVariableOp0batch_normalization_105/beta/Read/ReadVariableOp7batch_normalization_105/moving_mean/Read/ReadVariableOp;batch_normalization_105/moving_variance/Read/ReadVariableOp$conv2d_91/kernel/Read/ReadVariableOp"conv2d_91/bias/Read/ReadVariableOp1batch_normalization_106/gamma/Read/ReadVariableOp0batch_normalization_106/beta/Read/ReadVariableOp7batch_normalization_106/moving_mean/Read/ReadVariableOp;batch_normalization_106/moving_variance/Read/ReadVariableOp$conv2d_92/kernel/Read/ReadVariableOp"conv2d_92/bias/Read/ReadVariableOp1batch_normalization_107/gamma/Read/ReadVariableOp0batch_normalization_107/beta/Read/ReadVariableOp7batch_normalization_107/moving_mean/Read/ReadVariableOp;batch_normalization_107/moving_variance/Read/ReadVariableOp$conv2d_93/kernel/Read/ReadVariableOp"conv2d_93/bias/Read/ReadVariableOp1batch_normalization_108/gamma/Read/ReadVariableOp0batch_normalization_108/beta/Read/ReadVariableOp7batch_normalization_108/moving_mean/Read/ReadVariableOp;batch_normalization_108/moving_variance/Read/ReadVariableOp$conv2d_94/kernel/Read/ReadVariableOp"conv2d_94/bias/Read/ReadVariableOp1batch_normalization_109/gamma/Read/ReadVariableOp0batch_normalization_109/beta/Read/ReadVariableOp7batch_normalization_109/moving_mean/Read/ReadVariableOp;batch_normalization_109/moving_variance/Read/ReadVariableOp$conv2d_95/kernel/Read/ReadVariableOp"conv2d_95/bias/Read/ReadVariableOp1batch_normalization_110/gamma/Read/ReadVariableOp0batch_normalization_110/beta/Read/ReadVariableOp7batch_normalization_110/moving_mean/Read/ReadVariableOp;batch_normalization_110/moving_variance/Read/ReadVariableOp1batch_normalization_111/gamma/Read/ReadVariableOp0batch_normalization_111/beta/Read/ReadVariableOp7batch_normalization_111/moving_mean/Read/ReadVariableOp;batch_normalization_111/moving_variance/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOpConst*7
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
__inference__traced_save_221088
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_90/kernelconv2d_90/biasbatch_normalization_105/gammabatch_normalization_105/beta#batch_normalization_105/moving_mean'batch_normalization_105/moving_varianceconv2d_91/kernelconv2d_91/biasbatch_normalization_106/gammabatch_normalization_106/beta#batch_normalization_106/moving_mean'batch_normalization_106/moving_varianceconv2d_92/kernelconv2d_92/biasbatch_normalization_107/gammabatch_normalization_107/beta#batch_normalization_107/moving_mean'batch_normalization_107/moving_varianceconv2d_93/kernelconv2d_93/biasbatch_normalization_108/gammabatch_normalization_108/beta#batch_normalization_108/moving_mean'batch_normalization_108/moving_varianceconv2d_94/kernelconv2d_94/biasbatch_normalization_109/gammabatch_normalization_109/beta#batch_normalization_109/moving_mean'batch_normalization_109/moving_varianceconv2d_95/kernelconv2d_95/biasbatch_normalization_110/gammabatch_normalization_110/beta#batch_normalization_110/moving_mean'batch_normalization_110/moving_variancebatch_normalization_111/gammabatch_normalization_111/beta#batch_normalization_111/moving_mean'batch_normalization_111/moving_variancedense_16/kerneldense_16/bias*6
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
"__inference__traced_restore_221224��
�
�
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_217323

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
�
�
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_217543

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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219900

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
�
�
8__inference_batch_normalization_111_layer_call_fn_220863

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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_2179522
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
�j
�
I__inference_sequential_15_layer_call_and_return_conditional_losses_218968

inputs
conv2d_90_218864
conv2d_90_218866"
batch_normalization_105_218869"
batch_normalization_105_218871"
batch_normalization_105_218873"
batch_normalization_105_218875
conv2d_91_218878
conv2d_91_218880"
batch_normalization_106_218883"
batch_normalization_106_218885"
batch_normalization_106_218887"
batch_normalization_106_218889
conv2d_92_218893
conv2d_92_218895"
batch_normalization_107_218898"
batch_normalization_107_218900"
batch_normalization_107_218902"
batch_normalization_107_218904
conv2d_93_218907
conv2d_93_218909"
batch_normalization_108_218912"
batch_normalization_108_218914"
batch_normalization_108_218916"
batch_normalization_108_218918
conv2d_94_218922
conv2d_94_218924"
batch_normalization_109_218927"
batch_normalization_109_218929"
batch_normalization_109_218931"
batch_normalization_109_218933
conv2d_95_218936
conv2d_95_218938"
batch_normalization_110_218941"
batch_normalization_110_218943"
batch_normalization_110_218945"
batch_normalization_110_218947"
batch_normalization_111_218951"
batch_normalization_111_218953"
batch_normalization_111_218955"
batch_normalization_111_218957
dense_16_218961
dense_16_218963
identity��/batch_normalization_105/StatefulPartitionedCall�/batch_normalization_106/StatefulPartitionedCall�/batch_normalization_107/StatefulPartitionedCall�/batch_normalization_108/StatefulPartitionedCall�/batch_normalization_109/StatefulPartitionedCall�/batch_normalization_110/StatefulPartitionedCall�/batch_normalization_111/StatefulPartitionedCall�!conv2d_90/StatefulPartitionedCall�!conv2d_91/StatefulPartitionedCall�!conv2d_92/StatefulPartitionedCall�!conv2d_93/StatefulPartitionedCall�!conv2d_94/StatefulPartitionedCall�!conv2d_95/StatefulPartitionedCall� dense_16/StatefulPartitionedCall�
!conv2d_90/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_90_218864conv2d_90_218866*
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
E__inference_conv2d_90_layer_call_and_return_conditional_losses_2180092#
!conv2d_90/StatefulPartitionedCall�
/batch_normalization_105/StatefulPartitionedCallStatefulPartitionedCall*conv2d_90/StatefulPartitionedCall:output:0batch_normalization_105_218869batch_normalization_105_218871batch_normalization_105_218873batch_normalization_105_218875*
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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_21804421
/batch_normalization_105/StatefulPartitionedCall�
!conv2d_91/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_105/StatefulPartitionedCall:output:0conv2d_91_218878conv2d_91_218880*
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
E__inference_conv2d_91_layer_call_and_return_conditional_losses_2181092#
!conv2d_91/StatefulPartitionedCall�
/batch_normalization_106/StatefulPartitionedCallStatefulPartitionedCall*conv2d_91/StatefulPartitionedCall:output:0batch_normalization_106_218883batch_normalization_106_218885batch_normalization_106_218887batch_normalization_106_218889*
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
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_21814421
/batch_normalization_106/StatefulPartitionedCall�
 max_pooling2d_45/PartitionedCallPartitionedCall8batch_normalization_106/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_2174442"
 max_pooling2d_45/PartitionedCall�
!conv2d_92/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0conv2d_92_218893conv2d_92_218895*
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
E__inference_conv2d_92_layer_call_and_return_conditional_losses_2182102#
!conv2d_92/StatefulPartitionedCall�
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall*conv2d_92/StatefulPartitionedCall:output:0batch_normalization_107_218898batch_normalization_107_218900batch_normalization_107_218902batch_normalization_107_218904*
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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_21824521
/batch_normalization_107/StatefulPartitionedCall�
!conv2d_93/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0conv2d_93_218907conv2d_93_218909*
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
E__inference_conv2d_93_layer_call_and_return_conditional_losses_2183102#
!conv2d_93/StatefulPartitionedCall�
/batch_normalization_108/StatefulPartitionedCallStatefulPartitionedCall*conv2d_93/StatefulPartitionedCall:output:0batch_normalization_108_218912batch_normalization_108_218914batch_normalization_108_218916batch_normalization_108_218918*
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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_21834521
/batch_normalization_108/StatefulPartitionedCall�
 max_pooling2d_46/PartitionedCallPartitionedCall8batch_normalization_108/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_2176642"
 max_pooling2d_46/PartitionedCall�
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0conv2d_94_218922conv2d_94_218924*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_2184112#
!conv2d_94/StatefulPartitionedCall�
/batch_normalization_109/StatefulPartitionedCallStatefulPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0batch_normalization_109_218927batch_normalization_109_218929batch_normalization_109_218931batch_normalization_109_218933*
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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_21844621
/batch_normalization_109/StatefulPartitionedCall�
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_109/StatefulPartitionedCall:output:0conv2d_95_218936conv2d_95_218938*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_2185112#
!conv2d_95/StatefulPartitionedCall�
/batch_normalization_110/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_110_218941batch_normalization_110_218943batch_normalization_110_218945batch_normalization_110_218947*
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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_21854621
/batch_normalization_110/StatefulPartitionedCall�
 max_pooling2d_47/PartitionedCallPartitionedCall8batch_normalization_110/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2178842"
 max_pooling2d_47/PartitionedCall�
/batch_normalization_111/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0batch_normalization_111_218951batch_normalization_111_218953batch_normalization_111_218955batch_normalization_111_218957*
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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_21862021
/batch_normalization_111/StatefulPartitionedCall�
flatten_15/PartitionedCallPartitionedCall8batch_normalization_111/StatefulPartitionedCall:output:0*
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
F__inference_flatten_15_layer_call_and_return_conditional_losses_2186802
flatten_15/PartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_16_218961dense_16_218963*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_2186992"
 dense_16/StatefulPartitionedCall�
lambda_16/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
E__inference_lambda_16_layer_call_and_return_conditional_losses_2187262
lambda_16/PartitionedCall�
IdentityIdentity"lambda_16/PartitionedCall:output:00^batch_normalization_105/StatefulPartitionedCall0^batch_normalization_106/StatefulPartitionedCall0^batch_normalization_107/StatefulPartitionedCall0^batch_normalization_108/StatefulPartitionedCall0^batch_normalization_109/StatefulPartitionedCall0^batch_normalization_110/StatefulPartitionedCall0^batch_normalization_111/StatefulPartitionedCall"^conv2d_90/StatefulPartitionedCall"^conv2d_91/StatefulPartitionedCall"^conv2d_92/StatefulPartitionedCall"^conv2d_93/StatefulPartitionedCall"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2b
/batch_normalization_105/StatefulPartitionedCall/batch_normalization_105/StatefulPartitionedCall2b
/batch_normalization_106/StatefulPartitionedCall/batch_normalization_106/StatefulPartitionedCall2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2b
/batch_normalization_108/StatefulPartitionedCall/batch_normalization_108/StatefulPartitionedCall2b
/batch_normalization_109/StatefulPartitionedCall/batch_normalization_109/StatefulPartitionedCall2b
/batch_normalization_110/StatefulPartitionedCall/batch_normalization_110/StatefulPartitionedCall2b
/batch_normalization_111/StatefulPartitionedCall/batch_normalization_111/StatefulPartitionedCall2F
!conv2d_90/StatefulPartitionedCall!conv2d_90/StatefulPartitionedCall2F
!conv2d_91/StatefulPartitionedCall!conv2d_91/StatefulPartitionedCall2F
!conv2d_92/StatefulPartitionedCall!conv2d_92/StatefulPartitionedCall2F
!conv2d_93/StatefulPartitionedCall!conv2d_93/StatefulPartitionedCall2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_218345

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
h
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_217664

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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220492

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
�
D__inference_dense_16_layer_call_and_return_conditional_losses_220898

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
��
�"
I__inference_sequential_15_layer_call_and_return_conditional_losses_219682

inputs,
(conv2d_90_conv2d_readvariableop_resource-
)conv2d_90_biasadd_readvariableop_resource3
/batch_normalization_105_readvariableop_resource5
1batch_normalization_105_readvariableop_1_resourceD
@batch_normalization_105_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_105_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_91_conv2d_readvariableop_resource-
)conv2d_91_biasadd_readvariableop_resource3
/batch_normalization_106_readvariableop_resource5
1batch_normalization_106_readvariableop_1_resourceD
@batch_normalization_106_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_106_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_92_conv2d_readvariableop_resource-
)conv2d_92_biasadd_readvariableop_resource3
/batch_normalization_107_readvariableop_resource5
1batch_normalization_107_readvariableop_1_resourceD
@batch_normalization_107_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_107_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_93_conv2d_readvariableop_resource-
)conv2d_93_biasadd_readvariableop_resource3
/batch_normalization_108_readvariableop_resource5
1batch_normalization_108_readvariableop_1_resourceD
@batch_normalization_108_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_108_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_94_conv2d_readvariableop_resource-
)conv2d_94_biasadd_readvariableop_resource3
/batch_normalization_109_readvariableop_resource5
1batch_normalization_109_readvariableop_1_resourceD
@batch_normalization_109_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_109_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_95_conv2d_readvariableop_resource-
)conv2d_95_biasadd_readvariableop_resource3
/batch_normalization_110_readvariableop_resource5
1batch_normalization_110_readvariableop_1_resourceD
@batch_normalization_110_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_110_fusedbatchnormv3_readvariableop_1_resource3
/batch_normalization_111_readvariableop_resource5
1batch_normalization_111_readvariableop_1_resourceD
@batch_normalization_111_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_111_fusedbatchnormv3_readvariableop_1_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource
identity��7batch_normalization_105/FusedBatchNormV3/ReadVariableOp�9batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_105/ReadVariableOp�(batch_normalization_105/ReadVariableOp_1�7batch_normalization_106/FusedBatchNormV3/ReadVariableOp�9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_106/ReadVariableOp�(batch_normalization_106/ReadVariableOp_1�7batch_normalization_107/FusedBatchNormV3/ReadVariableOp�9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_107/ReadVariableOp�(batch_normalization_107/ReadVariableOp_1�7batch_normalization_108/FusedBatchNormV3/ReadVariableOp�9batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_108/ReadVariableOp�(batch_normalization_108/ReadVariableOp_1�7batch_normalization_109/FusedBatchNormV3/ReadVariableOp�9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_109/ReadVariableOp�(batch_normalization_109/ReadVariableOp_1�7batch_normalization_110/FusedBatchNormV3/ReadVariableOp�9batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_110/ReadVariableOp�(batch_normalization_110/ReadVariableOp_1�7batch_normalization_111/FusedBatchNormV3/ReadVariableOp�9batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_111/ReadVariableOp�(batch_normalization_111/ReadVariableOp_1� conv2d_90/BiasAdd/ReadVariableOp�conv2d_90/Conv2D/ReadVariableOp� conv2d_91/BiasAdd/ReadVariableOp�conv2d_91/Conv2D/ReadVariableOp� conv2d_92/BiasAdd/ReadVariableOp�conv2d_92/Conv2D/ReadVariableOp� conv2d_93/BiasAdd/ReadVariableOp�conv2d_93/Conv2D/ReadVariableOp� conv2d_94/BiasAdd/ReadVariableOp�conv2d_94/Conv2D/ReadVariableOp� conv2d_95/BiasAdd/ReadVariableOp�conv2d_95/Conv2D/ReadVariableOp�dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�
conv2d_90/Conv2D/ReadVariableOpReadVariableOp(conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_90/Conv2D/ReadVariableOp�
conv2d_90/Conv2DConv2Dinputs'conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_90/Conv2D�
 conv2d_90/BiasAdd/ReadVariableOpReadVariableOp)conv2d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_90/BiasAdd/ReadVariableOp�
conv2d_90/BiasAddBiasAddconv2d_90/Conv2D:output:0(conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_90/BiasAdd~
conv2d_90/ReluReluconv2d_90/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_90/Relu�
&batch_normalization_105/ReadVariableOpReadVariableOp/batch_normalization_105_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_105/ReadVariableOp�
(batch_normalization_105/ReadVariableOp_1ReadVariableOp1batch_normalization_105_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_105/ReadVariableOp_1�
7batch_normalization_105/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_105_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_105/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_105_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_105/FusedBatchNormV3FusedBatchNormV3conv2d_90/Relu:activations:0.batch_normalization_105/ReadVariableOp:value:00batch_normalization_105/ReadVariableOp_1:value:0?batch_normalization_105/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_105/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2*
(batch_normalization_105/FusedBatchNormV3�
conv2d_91/Conv2D/ReadVariableOpReadVariableOp(conv2d_91_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_91/Conv2D/ReadVariableOp�
conv2d_91/Conv2DConv2D,batch_normalization_105/FusedBatchNormV3:y:0'conv2d_91/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_91/Conv2D�
 conv2d_91/BiasAdd/ReadVariableOpReadVariableOp)conv2d_91_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_91/BiasAdd/ReadVariableOp�
conv2d_91/BiasAddBiasAddconv2d_91/Conv2D:output:0(conv2d_91/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_91/BiasAdd~
conv2d_91/ReluReluconv2d_91/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_91/Relu�
&batch_normalization_106/ReadVariableOpReadVariableOp/batch_normalization_106_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_106/ReadVariableOp�
(batch_normalization_106/ReadVariableOp_1ReadVariableOp1batch_normalization_106_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_106/ReadVariableOp_1�
7batch_normalization_106/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_106_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_106/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_106_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_106/FusedBatchNormV3FusedBatchNormV3conv2d_91/Relu:activations:0.batch_normalization_106/ReadVariableOp:value:00batch_normalization_106/ReadVariableOp_1:value:0?batch_normalization_106/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_106/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2*
(batch_normalization_106/FusedBatchNormV3�
max_pooling2d_45/MaxPoolMaxPool,batch_normalization_106/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_45/MaxPool�
conv2d_92/Conv2D/ReadVariableOpReadVariableOp(conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_92/Conv2D/ReadVariableOp�
conv2d_92/Conv2DConv2D!max_pooling2d_45/MaxPool:output:0'conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_92/Conv2D�
 conv2d_92/BiasAdd/ReadVariableOpReadVariableOp)conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_92/BiasAdd/ReadVariableOp�
conv2d_92/BiasAddBiasAddconv2d_92/Conv2D:output:0(conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_92/BiasAdd~
conv2d_92/ReluReluconv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_92/Relu�
&batch_normalization_107/ReadVariableOpReadVariableOp/batch_normalization_107_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_107/ReadVariableOp�
(batch_normalization_107/ReadVariableOp_1ReadVariableOp1batch_normalization_107_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_107/ReadVariableOp_1�
7batch_normalization_107/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_107_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_107/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_107_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_107/FusedBatchNormV3FusedBatchNormV3conv2d_92/Relu:activations:0.batch_normalization_107/ReadVariableOp:value:00batch_normalization_107/ReadVariableOp_1:value:0?batch_normalization_107/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_107/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2*
(batch_normalization_107/FusedBatchNormV3�
conv2d_93/Conv2D/ReadVariableOpReadVariableOp(conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_93/Conv2D/ReadVariableOp�
conv2d_93/Conv2DConv2D,batch_normalization_107/FusedBatchNormV3:y:0'conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_93/Conv2D�
 conv2d_93/BiasAdd/ReadVariableOpReadVariableOp)conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_93/BiasAdd/ReadVariableOp�
conv2d_93/BiasAddBiasAddconv2d_93/Conv2D:output:0(conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_93/BiasAdd~
conv2d_93/ReluReluconv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_93/Relu�
&batch_normalization_108/ReadVariableOpReadVariableOp/batch_normalization_108_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_108/ReadVariableOp�
(batch_normalization_108/ReadVariableOp_1ReadVariableOp1batch_normalization_108_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_108/ReadVariableOp_1�
7batch_normalization_108/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_108_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_108/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_108_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_108/FusedBatchNormV3FusedBatchNormV3conv2d_93/Relu:activations:0.batch_normalization_108/ReadVariableOp:value:00batch_normalization_108/ReadVariableOp_1:value:0?batch_normalization_108/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_108/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2*
(batch_normalization_108/FusedBatchNormV3�
max_pooling2d_46/MaxPoolMaxPool,batch_normalization_108/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_46/MaxPool�
conv2d_94/Conv2D/ReadVariableOpReadVariableOp(conv2d_94_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_94/Conv2D/ReadVariableOp�
conv2d_94/Conv2DConv2D!max_pooling2d_46/MaxPool:output:0'conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_94/Conv2D�
 conv2d_94/BiasAdd/ReadVariableOpReadVariableOp)conv2d_94_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_94/BiasAdd/ReadVariableOp�
conv2d_94/BiasAddBiasAddconv2d_94/Conv2D:output:0(conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_94/BiasAdd
conv2d_94/ReluReluconv2d_94/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_94/Relu�
&batch_normalization_109/ReadVariableOpReadVariableOp/batch_normalization_109_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_109/ReadVariableOp�
(batch_normalization_109/ReadVariableOp_1ReadVariableOp1batch_normalization_109_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_109/ReadVariableOp_1�
7batch_normalization_109/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_109_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_109/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_109_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_109/FusedBatchNormV3FusedBatchNormV3conv2d_94/Relu:activations:0.batch_normalization_109/ReadVariableOp:value:00batch_normalization_109/ReadVariableOp_1:value:0?batch_normalization_109/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_109/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2*
(batch_normalization_109/FusedBatchNormV3�
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_95/Conv2D/ReadVariableOp�
conv2d_95/Conv2DConv2D,batch_normalization_109/FusedBatchNormV3:y:0'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_95/Conv2D�
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp�
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_95/BiasAdd
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_95/Relu�
&batch_normalization_110/ReadVariableOpReadVariableOp/batch_normalization_110_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_110/ReadVariableOp�
(batch_normalization_110/ReadVariableOp_1ReadVariableOp1batch_normalization_110_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_110/ReadVariableOp_1�
7batch_normalization_110/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_110_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_110/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_110_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_110/FusedBatchNormV3FusedBatchNormV3conv2d_95/Relu:activations:0.batch_normalization_110/ReadVariableOp:value:00batch_normalization_110/ReadVariableOp_1:value:0?batch_normalization_110/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_110/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2*
(batch_normalization_110/FusedBatchNormV3�
max_pooling2d_47/MaxPoolMaxPool,batch_normalization_110/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_47/MaxPool�
&batch_normalization_111/ReadVariableOpReadVariableOp/batch_normalization_111_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_111/ReadVariableOp�
(batch_normalization_111/ReadVariableOp_1ReadVariableOp1batch_normalization_111_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_111/ReadVariableOp_1�
7batch_normalization_111/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_111_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_111/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_111_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_111/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_47/MaxPool:output:0.batch_normalization_111/ReadVariableOp:value:00batch_normalization_111/ReadVariableOp_1:value:0?batch_normalization_111/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_111/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2*
(batch_normalization_111/FusedBatchNormV3u
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_15/Const�
flatten_15/ReshapeReshape,batch_normalization_111/FusedBatchNormV3:y:0flatten_15/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_15/Reshape�
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_16/MatMul/ReadVariableOp�
dense_16/MatMulMatMulflatten_15/Reshape:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_16/MatMul�
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_16/BiasAdd/ReadVariableOp�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_16/BiasAdd}
dense_16/SigmoidSigmoiddense_16/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_16/Sigmoid�
lambda_16/l2_normalize/SquareSquaredense_16/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
lambda_16/l2_normalize/Square�
,lambda_16/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,lambda_16/l2_normalize/Sum/reduction_indices�
lambda_16/l2_normalize/SumSum!lambda_16/l2_normalize/Square:y:05lambda_16/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
lambda_16/l2_normalize/Sum�
 lambda_16/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2"
 lambda_16/l2_normalize/Maximum/y�
lambda_16/l2_normalize/MaximumMaximum#lambda_16/l2_normalize/Sum:output:0)lambda_16/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2 
lambda_16/l2_normalize/Maximum�
lambda_16/l2_normalize/RsqrtRsqrt"lambda_16/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
lambda_16/l2_normalize/Rsqrt�
lambda_16/l2_normalizeMuldense_16/Sigmoid:y:0 lambda_16/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
lambda_16/l2_normalize�
IdentityIdentitylambda_16/l2_normalize:z:08^batch_normalization_105/FusedBatchNormV3/ReadVariableOp:^batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_105/ReadVariableOp)^batch_normalization_105/ReadVariableOp_18^batch_normalization_106/FusedBatchNormV3/ReadVariableOp:^batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_106/ReadVariableOp)^batch_normalization_106/ReadVariableOp_18^batch_normalization_107/FusedBatchNormV3/ReadVariableOp:^batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_107/ReadVariableOp)^batch_normalization_107/ReadVariableOp_18^batch_normalization_108/FusedBatchNormV3/ReadVariableOp:^batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_108/ReadVariableOp)^batch_normalization_108/ReadVariableOp_18^batch_normalization_109/FusedBatchNormV3/ReadVariableOp:^batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_109/ReadVariableOp)^batch_normalization_109/ReadVariableOp_18^batch_normalization_110/FusedBatchNormV3/ReadVariableOp:^batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_110/ReadVariableOp)^batch_normalization_110/ReadVariableOp_18^batch_normalization_111/FusedBatchNormV3/ReadVariableOp:^batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_111/ReadVariableOp)^batch_normalization_111/ReadVariableOp_1!^conv2d_90/BiasAdd/ReadVariableOp ^conv2d_90/Conv2D/ReadVariableOp!^conv2d_91/BiasAdd/ReadVariableOp ^conv2d_91/Conv2D/ReadVariableOp!^conv2d_92/BiasAdd/ReadVariableOp ^conv2d_92/Conv2D/ReadVariableOp!^conv2d_93/BiasAdd/ReadVariableOp ^conv2d_93/Conv2D/ReadVariableOp!^conv2d_94/BiasAdd/ReadVariableOp ^conv2d_94/Conv2D/ReadVariableOp!^conv2d_95/BiasAdd/ReadVariableOp ^conv2d_95/Conv2D/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2r
7batch_normalization_105/FusedBatchNormV3/ReadVariableOp7batch_normalization_105/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_105/FusedBatchNormV3/ReadVariableOp_19batch_normalization_105/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_105/ReadVariableOp&batch_normalization_105/ReadVariableOp2T
(batch_normalization_105/ReadVariableOp_1(batch_normalization_105/ReadVariableOp_12r
7batch_normalization_106/FusedBatchNormV3/ReadVariableOp7batch_normalization_106/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_19batch_normalization_106/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_106/ReadVariableOp&batch_normalization_106/ReadVariableOp2T
(batch_normalization_106/ReadVariableOp_1(batch_normalization_106/ReadVariableOp_12r
7batch_normalization_107/FusedBatchNormV3/ReadVariableOp7batch_normalization_107/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_19batch_normalization_107/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_107/ReadVariableOp&batch_normalization_107/ReadVariableOp2T
(batch_normalization_107/ReadVariableOp_1(batch_normalization_107/ReadVariableOp_12r
7batch_normalization_108/FusedBatchNormV3/ReadVariableOp7batch_normalization_108/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_108/FusedBatchNormV3/ReadVariableOp_19batch_normalization_108/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_108/ReadVariableOp&batch_normalization_108/ReadVariableOp2T
(batch_normalization_108/ReadVariableOp_1(batch_normalization_108/ReadVariableOp_12r
7batch_normalization_109/FusedBatchNormV3/ReadVariableOp7batch_normalization_109/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_19batch_normalization_109/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_109/ReadVariableOp&batch_normalization_109/ReadVariableOp2T
(batch_normalization_109/ReadVariableOp_1(batch_normalization_109/ReadVariableOp_12r
7batch_normalization_110/FusedBatchNormV3/ReadVariableOp7batch_normalization_110/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_110/FusedBatchNormV3/ReadVariableOp_19batch_normalization_110/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_110/ReadVariableOp&batch_normalization_110/ReadVariableOp2T
(batch_normalization_110/ReadVariableOp_1(batch_normalization_110/ReadVariableOp_12r
7batch_normalization_111/FusedBatchNormV3/ReadVariableOp7batch_normalization_111/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_111/FusedBatchNormV3/ReadVariableOp_19batch_normalization_111/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_111/ReadVariableOp&batch_normalization_111/ReadVariableOp2T
(batch_normalization_111/ReadVariableOp_1(batch_normalization_111/ReadVariableOp_12D
 conv2d_90/BiasAdd/ReadVariableOp conv2d_90/BiasAdd/ReadVariableOp2B
conv2d_90/Conv2D/ReadVariableOpconv2d_90/Conv2D/ReadVariableOp2D
 conv2d_91/BiasAdd/ReadVariableOp conv2d_91/BiasAdd/ReadVariableOp2B
conv2d_91/Conv2D/ReadVariableOpconv2d_91/Conv2D/ReadVariableOp2D
 conv2d_92/BiasAdd/ReadVariableOp conv2d_92/BiasAdd/ReadVariableOp2B
conv2d_92/Conv2D/ReadVariableOpconv2d_92/Conv2D/ReadVariableOp2D
 conv2d_93/BiasAdd/ReadVariableOp conv2d_93/BiasAdd/ReadVariableOp2B
conv2d_93/Conv2D/ReadVariableOpconv2d_93/Conv2D/ReadVariableOp2D
 conv2d_94/BiasAdd/ReadVariableOp conv2d_94/BiasAdd/ReadVariableOp2B
conv2d_94/Conv2D/ReadVariableOpconv2d_94/Conv2D/ReadVariableOp2D
 conv2d_95/BiasAdd/ReadVariableOp conv2d_95/BiasAdd/ReadVariableOp2B
conv2d_95/Conv2D/ReadVariableOpconv2d_95/Conv2D/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�k
�
I__inference_sequential_15_layer_call_and_return_conditional_losses_218858
conv2d_90_input
conv2d_90_218754
conv2d_90_218756"
batch_normalization_105_218759"
batch_normalization_105_218761"
batch_normalization_105_218763"
batch_normalization_105_218765
conv2d_91_218768
conv2d_91_218770"
batch_normalization_106_218773"
batch_normalization_106_218775"
batch_normalization_106_218777"
batch_normalization_106_218779
conv2d_92_218783
conv2d_92_218785"
batch_normalization_107_218788"
batch_normalization_107_218790"
batch_normalization_107_218792"
batch_normalization_107_218794
conv2d_93_218797
conv2d_93_218799"
batch_normalization_108_218802"
batch_normalization_108_218804"
batch_normalization_108_218806"
batch_normalization_108_218808
conv2d_94_218812
conv2d_94_218814"
batch_normalization_109_218817"
batch_normalization_109_218819"
batch_normalization_109_218821"
batch_normalization_109_218823
conv2d_95_218826
conv2d_95_218828"
batch_normalization_110_218831"
batch_normalization_110_218833"
batch_normalization_110_218835"
batch_normalization_110_218837"
batch_normalization_111_218841"
batch_normalization_111_218843"
batch_normalization_111_218845"
batch_normalization_111_218847
dense_16_218851
dense_16_218853
identity��/batch_normalization_105/StatefulPartitionedCall�/batch_normalization_106/StatefulPartitionedCall�/batch_normalization_107/StatefulPartitionedCall�/batch_normalization_108/StatefulPartitionedCall�/batch_normalization_109/StatefulPartitionedCall�/batch_normalization_110/StatefulPartitionedCall�/batch_normalization_111/StatefulPartitionedCall�!conv2d_90/StatefulPartitionedCall�!conv2d_91/StatefulPartitionedCall�!conv2d_92/StatefulPartitionedCall�!conv2d_93/StatefulPartitionedCall�!conv2d_94/StatefulPartitionedCall�!conv2d_95/StatefulPartitionedCall� dense_16/StatefulPartitionedCall�
!conv2d_90/StatefulPartitionedCallStatefulPartitionedCallconv2d_90_inputconv2d_90_218754conv2d_90_218756*
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
E__inference_conv2d_90_layer_call_and_return_conditional_losses_2180092#
!conv2d_90/StatefulPartitionedCall�
/batch_normalization_105/StatefulPartitionedCallStatefulPartitionedCall*conv2d_90/StatefulPartitionedCall:output:0batch_normalization_105_218759batch_normalization_105_218761batch_normalization_105_218763batch_normalization_105_218765*
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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_21806221
/batch_normalization_105/StatefulPartitionedCall�
!conv2d_91/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_105/StatefulPartitionedCall:output:0conv2d_91_218768conv2d_91_218770*
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
E__inference_conv2d_91_layer_call_and_return_conditional_losses_2181092#
!conv2d_91/StatefulPartitionedCall�
/batch_normalization_106/StatefulPartitionedCallStatefulPartitionedCall*conv2d_91/StatefulPartitionedCall:output:0batch_normalization_106_218773batch_normalization_106_218775batch_normalization_106_218777batch_normalization_106_218779*
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
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_21816221
/batch_normalization_106/StatefulPartitionedCall�
 max_pooling2d_45/PartitionedCallPartitionedCall8batch_normalization_106/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_2174442"
 max_pooling2d_45/PartitionedCall�
!conv2d_92/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0conv2d_92_218783conv2d_92_218785*
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
E__inference_conv2d_92_layer_call_and_return_conditional_losses_2182102#
!conv2d_92/StatefulPartitionedCall�
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall*conv2d_92/StatefulPartitionedCall:output:0batch_normalization_107_218788batch_normalization_107_218790batch_normalization_107_218792batch_normalization_107_218794*
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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_21826321
/batch_normalization_107/StatefulPartitionedCall�
!conv2d_93/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0conv2d_93_218797conv2d_93_218799*
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
E__inference_conv2d_93_layer_call_and_return_conditional_losses_2183102#
!conv2d_93/StatefulPartitionedCall�
/batch_normalization_108/StatefulPartitionedCallStatefulPartitionedCall*conv2d_93/StatefulPartitionedCall:output:0batch_normalization_108_218802batch_normalization_108_218804batch_normalization_108_218806batch_normalization_108_218808*
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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_21836321
/batch_normalization_108/StatefulPartitionedCall�
 max_pooling2d_46/PartitionedCallPartitionedCall8batch_normalization_108/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_2176642"
 max_pooling2d_46/PartitionedCall�
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0conv2d_94_218812conv2d_94_218814*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_2184112#
!conv2d_94/StatefulPartitionedCall�
/batch_normalization_109/StatefulPartitionedCallStatefulPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0batch_normalization_109_218817batch_normalization_109_218819batch_normalization_109_218821batch_normalization_109_218823*
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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_21846421
/batch_normalization_109/StatefulPartitionedCall�
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_109/StatefulPartitionedCall:output:0conv2d_95_218826conv2d_95_218828*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_2185112#
!conv2d_95/StatefulPartitionedCall�
/batch_normalization_110/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_110_218831batch_normalization_110_218833batch_normalization_110_218835batch_normalization_110_218837*
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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_21856421
/batch_normalization_110/StatefulPartitionedCall�
 max_pooling2d_47/PartitionedCallPartitionedCall8batch_normalization_110/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2178842"
 max_pooling2d_47/PartitionedCall�
/batch_normalization_111/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0batch_normalization_111_218841batch_normalization_111_218843batch_normalization_111_218845batch_normalization_111_218847*
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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_21863821
/batch_normalization_111/StatefulPartitionedCall�
flatten_15/PartitionedCallPartitionedCall8batch_normalization_111/StatefulPartitionedCall:output:0*
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
F__inference_flatten_15_layer_call_and_return_conditional_losses_2186802
flatten_15/PartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_16_218851dense_16_218853*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_2186992"
 dense_16/StatefulPartitionedCall�
lambda_16/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
E__inference_lambda_16_layer_call_and_return_conditional_losses_2187372
lambda_16/PartitionedCall�
IdentityIdentity"lambda_16/PartitionedCall:output:00^batch_normalization_105/StatefulPartitionedCall0^batch_normalization_106/StatefulPartitionedCall0^batch_normalization_107/StatefulPartitionedCall0^batch_normalization_108/StatefulPartitionedCall0^batch_normalization_109/StatefulPartitionedCall0^batch_normalization_110/StatefulPartitionedCall0^batch_normalization_111/StatefulPartitionedCall"^conv2d_90/StatefulPartitionedCall"^conv2d_91/StatefulPartitionedCall"^conv2d_92/StatefulPartitionedCall"^conv2d_93/StatefulPartitionedCall"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2b
/batch_normalization_105/StatefulPartitionedCall/batch_normalization_105/StatefulPartitionedCall2b
/batch_normalization_106/StatefulPartitionedCall/batch_normalization_106/StatefulPartitionedCall2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2b
/batch_normalization_108/StatefulPartitionedCall/batch_normalization_108/StatefulPartitionedCall2b
/batch_normalization_109/StatefulPartitionedCall/batch_normalization_109/StatefulPartitionedCall2b
/batch_normalization_110/StatefulPartitionedCall/batch_normalization_110/StatefulPartitionedCall2b
/batch_normalization_111/StatefulPartitionedCall/batch_normalization_111/StatefulPartitionedCall2F
!conv2d_90/StatefulPartitionedCall!conv2d_90/StatefulPartitionedCall2F
!conv2d_91/StatefulPartitionedCall!conv2d_91/StatefulPartitionedCall2F
!conv2d_92/StatefulPartitionedCall!conv2d_92/StatefulPartitionedCall2F
!conv2d_93/StatefulPartitionedCall!conv2d_93/StatefulPartitionedCall2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_90_input
�

�
E__inference_conv2d_91_layer_call_and_return_conditional_losses_218109

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
�
h
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_217884

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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_217292

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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_218245

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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220704

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
�
�
8__inference_batch_normalization_110_layer_call_fn_220748

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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_2185642
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
�
�
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_217983

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
E__inference_lambda_16_layer_call_and_return_conditional_losses_218726

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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_218638

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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220850

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

�
E__inference_conv2d_95_layer_call_and_return_conditional_losses_218511

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
�k
�
I__inference_sequential_15_layer_call_and_return_conditional_losses_218751
conv2d_90_input
conv2d_90_218020
conv2d_90_218022"
batch_normalization_105_218089"
batch_normalization_105_218091"
batch_normalization_105_218093"
batch_normalization_105_218095
conv2d_91_218120
conv2d_91_218122"
batch_normalization_106_218189"
batch_normalization_106_218191"
batch_normalization_106_218193"
batch_normalization_106_218195
conv2d_92_218221
conv2d_92_218223"
batch_normalization_107_218290"
batch_normalization_107_218292"
batch_normalization_107_218294"
batch_normalization_107_218296
conv2d_93_218321
conv2d_93_218323"
batch_normalization_108_218390"
batch_normalization_108_218392"
batch_normalization_108_218394"
batch_normalization_108_218396
conv2d_94_218422
conv2d_94_218424"
batch_normalization_109_218491"
batch_normalization_109_218493"
batch_normalization_109_218495"
batch_normalization_109_218497
conv2d_95_218522
conv2d_95_218524"
batch_normalization_110_218591"
batch_normalization_110_218593"
batch_normalization_110_218595"
batch_normalization_110_218597"
batch_normalization_111_218665"
batch_normalization_111_218667"
batch_normalization_111_218669"
batch_normalization_111_218671
dense_16_218710
dense_16_218712
identity��/batch_normalization_105/StatefulPartitionedCall�/batch_normalization_106/StatefulPartitionedCall�/batch_normalization_107/StatefulPartitionedCall�/batch_normalization_108/StatefulPartitionedCall�/batch_normalization_109/StatefulPartitionedCall�/batch_normalization_110/StatefulPartitionedCall�/batch_normalization_111/StatefulPartitionedCall�!conv2d_90/StatefulPartitionedCall�!conv2d_91/StatefulPartitionedCall�!conv2d_92/StatefulPartitionedCall�!conv2d_93/StatefulPartitionedCall�!conv2d_94/StatefulPartitionedCall�!conv2d_95/StatefulPartitionedCall� dense_16/StatefulPartitionedCall�
!conv2d_90/StatefulPartitionedCallStatefulPartitionedCallconv2d_90_inputconv2d_90_218020conv2d_90_218022*
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
E__inference_conv2d_90_layer_call_and_return_conditional_losses_2180092#
!conv2d_90/StatefulPartitionedCall�
/batch_normalization_105/StatefulPartitionedCallStatefulPartitionedCall*conv2d_90/StatefulPartitionedCall:output:0batch_normalization_105_218089batch_normalization_105_218091batch_normalization_105_218093batch_normalization_105_218095*
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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_21804421
/batch_normalization_105/StatefulPartitionedCall�
!conv2d_91/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_105/StatefulPartitionedCall:output:0conv2d_91_218120conv2d_91_218122*
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
E__inference_conv2d_91_layer_call_and_return_conditional_losses_2181092#
!conv2d_91/StatefulPartitionedCall�
/batch_normalization_106/StatefulPartitionedCallStatefulPartitionedCall*conv2d_91/StatefulPartitionedCall:output:0batch_normalization_106_218189batch_normalization_106_218191batch_normalization_106_218193batch_normalization_106_218195*
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
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_21814421
/batch_normalization_106/StatefulPartitionedCall�
 max_pooling2d_45/PartitionedCallPartitionedCall8batch_normalization_106/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_2174442"
 max_pooling2d_45/PartitionedCall�
!conv2d_92/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0conv2d_92_218221conv2d_92_218223*
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
E__inference_conv2d_92_layer_call_and_return_conditional_losses_2182102#
!conv2d_92/StatefulPartitionedCall�
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall*conv2d_92/StatefulPartitionedCall:output:0batch_normalization_107_218290batch_normalization_107_218292batch_normalization_107_218294batch_normalization_107_218296*
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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_21824521
/batch_normalization_107/StatefulPartitionedCall�
!conv2d_93/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0conv2d_93_218321conv2d_93_218323*
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
E__inference_conv2d_93_layer_call_and_return_conditional_losses_2183102#
!conv2d_93/StatefulPartitionedCall�
/batch_normalization_108/StatefulPartitionedCallStatefulPartitionedCall*conv2d_93/StatefulPartitionedCall:output:0batch_normalization_108_218390batch_normalization_108_218392batch_normalization_108_218394batch_normalization_108_218396*
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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_21834521
/batch_normalization_108/StatefulPartitionedCall�
 max_pooling2d_46/PartitionedCallPartitionedCall8batch_normalization_108/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_2176642"
 max_pooling2d_46/PartitionedCall�
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0conv2d_94_218422conv2d_94_218424*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_2184112#
!conv2d_94/StatefulPartitionedCall�
/batch_normalization_109/StatefulPartitionedCallStatefulPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0batch_normalization_109_218491batch_normalization_109_218493batch_normalization_109_218495batch_normalization_109_218497*
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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_21844621
/batch_normalization_109/StatefulPartitionedCall�
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_109/StatefulPartitionedCall:output:0conv2d_95_218522conv2d_95_218524*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_2185112#
!conv2d_95/StatefulPartitionedCall�
/batch_normalization_110/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_110_218591batch_normalization_110_218593batch_normalization_110_218595batch_normalization_110_218597*
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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_21854621
/batch_normalization_110/StatefulPartitionedCall�
 max_pooling2d_47/PartitionedCallPartitionedCall8batch_normalization_110/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2178842"
 max_pooling2d_47/PartitionedCall�
/batch_normalization_111/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0batch_normalization_111_218665batch_normalization_111_218667batch_normalization_111_218669batch_normalization_111_218671*
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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_21862021
/batch_normalization_111/StatefulPartitionedCall�
flatten_15/PartitionedCallPartitionedCall8batch_normalization_111/StatefulPartitionedCall:output:0*
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
F__inference_flatten_15_layer_call_and_return_conditional_losses_2186802
flatten_15/PartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_16_218710dense_16_218712*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_2186992"
 dense_16/StatefulPartitionedCall�
lambda_16/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
E__inference_lambda_16_layer_call_and_return_conditional_losses_2187262
lambda_16/PartitionedCall�
IdentityIdentity"lambda_16/PartitionedCall:output:00^batch_normalization_105/StatefulPartitionedCall0^batch_normalization_106/StatefulPartitionedCall0^batch_normalization_107/StatefulPartitionedCall0^batch_normalization_108/StatefulPartitionedCall0^batch_normalization_109/StatefulPartitionedCall0^batch_normalization_110/StatefulPartitionedCall0^batch_normalization_111/StatefulPartitionedCall"^conv2d_90/StatefulPartitionedCall"^conv2d_91/StatefulPartitionedCall"^conv2d_92/StatefulPartitionedCall"^conv2d_93/StatefulPartitionedCall"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2b
/batch_normalization_105/StatefulPartitionedCall/batch_normalization_105/StatefulPartitionedCall2b
/batch_normalization_106/StatefulPartitionedCall/batch_normalization_106/StatefulPartitionedCall2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2b
/batch_normalization_108/StatefulPartitionedCall/batch_normalization_108/StatefulPartitionedCall2b
/batch_normalization_109/StatefulPartitionedCall/batch_normalization_109/StatefulPartitionedCall2b
/batch_normalization_110/StatefulPartitionedCall/batch_normalization_110/StatefulPartitionedCall2b
/batch_normalization_111/StatefulPartitionedCall/batch_normalization_111/StatefulPartitionedCall2F
!conv2d_90/StatefulPartitionedCall!conv2d_90/StatefulPartitionedCall2F
!conv2d_91/StatefulPartitionedCall!conv2d_91/StatefulPartitionedCall2F
!conv2d_92/StatefulPartitionedCall!conv2d_92/StatefulPartitionedCall2F
!conv2d_93/StatefulPartitionedCall!conv2d_93/StatefulPartitionedCall2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_90_input
�
�
8__inference_batch_normalization_109_layer_call_fn_220536

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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_2177632
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
�
�
.__inference_sequential_15_layer_call_fn_219771

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
I__inference_sequential_15_layer_call_and_return_conditional_losses_2189682
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
�
�
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_218144

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
�
�
8__inference_batch_normalization_106_layer_call_fn_220092

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
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_2174272
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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220556

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
�
�
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220214

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
�
�
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219918

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
8__inference_batch_normalization_109_layer_call_fn_220587

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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_2184462
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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_217952

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
E__inference_lambda_16_layer_call_and_return_conditional_losses_220929

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
�
�
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_218263

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
�
�
8__inference_batch_normalization_106_layer_call_fn_220143

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
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_2181442
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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_218044

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
�
�
8__inference_batch_normalization_106_layer_call_fn_220156

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
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_2181622
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
�

�
E__inference_conv2d_92_layer_call_and_return_conditional_losses_218210

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
�
�
8__inference_batch_normalization_108_layer_call_fn_220452

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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_2176472
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
У
�&
I__inference_sequential_15_layer_call_and_return_conditional_losses_219519

inputs,
(conv2d_90_conv2d_readvariableop_resource-
)conv2d_90_biasadd_readvariableop_resource3
/batch_normalization_105_readvariableop_resource5
1batch_normalization_105_readvariableop_1_resourceD
@batch_normalization_105_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_105_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_91_conv2d_readvariableop_resource-
)conv2d_91_biasadd_readvariableop_resource3
/batch_normalization_106_readvariableop_resource5
1batch_normalization_106_readvariableop_1_resourceD
@batch_normalization_106_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_106_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_92_conv2d_readvariableop_resource-
)conv2d_92_biasadd_readvariableop_resource3
/batch_normalization_107_readvariableop_resource5
1batch_normalization_107_readvariableop_1_resourceD
@batch_normalization_107_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_107_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_93_conv2d_readvariableop_resource-
)conv2d_93_biasadd_readvariableop_resource3
/batch_normalization_108_readvariableop_resource5
1batch_normalization_108_readvariableop_1_resourceD
@batch_normalization_108_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_108_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_94_conv2d_readvariableop_resource-
)conv2d_94_biasadd_readvariableop_resource3
/batch_normalization_109_readvariableop_resource5
1batch_normalization_109_readvariableop_1_resourceD
@batch_normalization_109_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_109_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_95_conv2d_readvariableop_resource-
)conv2d_95_biasadd_readvariableop_resource3
/batch_normalization_110_readvariableop_resource5
1batch_normalization_110_readvariableop_1_resourceD
@batch_normalization_110_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_110_fusedbatchnormv3_readvariableop_1_resource3
/batch_normalization_111_readvariableop_resource5
1batch_normalization_111_readvariableop_1_resourceD
@batch_normalization_111_fusedbatchnormv3_readvariableop_resourceF
Bbatch_normalization_111_fusedbatchnormv3_readvariableop_1_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource
identity��&batch_normalization_105/AssignNewValue�(batch_normalization_105/AssignNewValue_1�7batch_normalization_105/FusedBatchNormV3/ReadVariableOp�9batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_105/ReadVariableOp�(batch_normalization_105/ReadVariableOp_1�&batch_normalization_106/AssignNewValue�(batch_normalization_106/AssignNewValue_1�7batch_normalization_106/FusedBatchNormV3/ReadVariableOp�9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_106/ReadVariableOp�(batch_normalization_106/ReadVariableOp_1�&batch_normalization_107/AssignNewValue�(batch_normalization_107/AssignNewValue_1�7batch_normalization_107/FusedBatchNormV3/ReadVariableOp�9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_107/ReadVariableOp�(batch_normalization_107/ReadVariableOp_1�&batch_normalization_108/AssignNewValue�(batch_normalization_108/AssignNewValue_1�7batch_normalization_108/FusedBatchNormV3/ReadVariableOp�9batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_108/ReadVariableOp�(batch_normalization_108/ReadVariableOp_1�&batch_normalization_109/AssignNewValue�(batch_normalization_109/AssignNewValue_1�7batch_normalization_109/FusedBatchNormV3/ReadVariableOp�9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_109/ReadVariableOp�(batch_normalization_109/ReadVariableOp_1�&batch_normalization_110/AssignNewValue�(batch_normalization_110/AssignNewValue_1�7batch_normalization_110/FusedBatchNormV3/ReadVariableOp�9batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_110/ReadVariableOp�(batch_normalization_110/ReadVariableOp_1�&batch_normalization_111/AssignNewValue�(batch_normalization_111/AssignNewValue_1�7batch_normalization_111/FusedBatchNormV3/ReadVariableOp�9batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1�&batch_normalization_111/ReadVariableOp�(batch_normalization_111/ReadVariableOp_1� conv2d_90/BiasAdd/ReadVariableOp�conv2d_90/Conv2D/ReadVariableOp� conv2d_91/BiasAdd/ReadVariableOp�conv2d_91/Conv2D/ReadVariableOp� conv2d_92/BiasAdd/ReadVariableOp�conv2d_92/Conv2D/ReadVariableOp� conv2d_93/BiasAdd/ReadVariableOp�conv2d_93/Conv2D/ReadVariableOp� conv2d_94/BiasAdd/ReadVariableOp�conv2d_94/Conv2D/ReadVariableOp� conv2d_95/BiasAdd/ReadVariableOp�conv2d_95/Conv2D/ReadVariableOp�dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�
conv2d_90/Conv2D/ReadVariableOpReadVariableOp(conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_90/Conv2D/ReadVariableOp�
conv2d_90/Conv2DConv2Dinputs'conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_90/Conv2D�
 conv2d_90/BiasAdd/ReadVariableOpReadVariableOp)conv2d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_90/BiasAdd/ReadVariableOp�
conv2d_90/BiasAddBiasAddconv2d_90/Conv2D:output:0(conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_90/BiasAdd~
conv2d_90/ReluReluconv2d_90/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_90/Relu�
&batch_normalization_105/ReadVariableOpReadVariableOp/batch_normalization_105_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_105/ReadVariableOp�
(batch_normalization_105/ReadVariableOp_1ReadVariableOp1batch_normalization_105_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_105/ReadVariableOp_1�
7batch_normalization_105/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_105_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_105/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_105_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_105/FusedBatchNormV3FusedBatchNormV3conv2d_90/Relu:activations:0.batch_normalization_105/ReadVariableOp:value:00batch_normalization_105/ReadVariableOp_1:value:0?batch_normalization_105/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_105/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_105/FusedBatchNormV3�
&batch_normalization_105/AssignNewValueAssignVariableOp@batch_normalization_105_fusedbatchnormv3_readvariableop_resource5batch_normalization_105/FusedBatchNormV3:batch_mean:08^batch_normalization_105/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_105/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_105/AssignNewValue�
(batch_normalization_105/AssignNewValue_1AssignVariableOpBbatch_normalization_105_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_105/FusedBatchNormV3:batch_variance:0:^batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_105/AssignNewValue_1�
conv2d_91/Conv2D/ReadVariableOpReadVariableOp(conv2d_91_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_91/Conv2D/ReadVariableOp�
conv2d_91/Conv2DConv2D,batch_normalization_105/FusedBatchNormV3:y:0'conv2d_91/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_91/Conv2D�
 conv2d_91/BiasAdd/ReadVariableOpReadVariableOp)conv2d_91_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_91/BiasAdd/ReadVariableOp�
conv2d_91/BiasAddBiasAddconv2d_91/Conv2D:output:0(conv2d_91/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_91/BiasAdd~
conv2d_91/ReluReluconv2d_91/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_91/Relu�
&batch_normalization_106/ReadVariableOpReadVariableOp/batch_normalization_106_readvariableop_resource*
_output_shapes
: *
dtype02(
&batch_normalization_106/ReadVariableOp�
(batch_normalization_106/ReadVariableOp_1ReadVariableOp1batch_normalization_106_readvariableop_1_resource*
_output_shapes
: *
dtype02*
(batch_normalization_106/ReadVariableOp_1�
7batch_normalization_106/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_106_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_106/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_106_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_106/FusedBatchNormV3FusedBatchNormV3conv2d_91/Relu:activations:0.batch_normalization_106/ReadVariableOp:value:00batch_normalization_106/ReadVariableOp_1:value:0?batch_normalization_106/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_106/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_106/FusedBatchNormV3�
&batch_normalization_106/AssignNewValueAssignVariableOp@batch_normalization_106_fusedbatchnormv3_readvariableop_resource5batch_normalization_106/FusedBatchNormV3:batch_mean:08^batch_normalization_106/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_106/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_106/AssignNewValue�
(batch_normalization_106/AssignNewValue_1AssignVariableOpBbatch_normalization_106_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_106/FusedBatchNormV3:batch_variance:0:^batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_106/AssignNewValue_1�
max_pooling2d_45/MaxPoolMaxPool,batch_normalization_106/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_45/MaxPool�
conv2d_92/Conv2D/ReadVariableOpReadVariableOp(conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_92/Conv2D/ReadVariableOp�
conv2d_92/Conv2DConv2D!max_pooling2d_45/MaxPool:output:0'conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_92/Conv2D�
 conv2d_92/BiasAdd/ReadVariableOpReadVariableOp)conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_92/BiasAdd/ReadVariableOp�
conv2d_92/BiasAddBiasAddconv2d_92/Conv2D:output:0(conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_92/BiasAdd~
conv2d_92/ReluReluconv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_92/Relu�
&batch_normalization_107/ReadVariableOpReadVariableOp/batch_normalization_107_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_107/ReadVariableOp�
(batch_normalization_107/ReadVariableOp_1ReadVariableOp1batch_normalization_107_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_107/ReadVariableOp_1�
7batch_normalization_107/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_107_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_107/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_107_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_107/FusedBatchNormV3FusedBatchNormV3conv2d_92/Relu:activations:0.batch_normalization_107/ReadVariableOp:value:00batch_normalization_107/ReadVariableOp_1:value:0?batch_normalization_107/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_107/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_107/FusedBatchNormV3�
&batch_normalization_107/AssignNewValueAssignVariableOp@batch_normalization_107_fusedbatchnormv3_readvariableop_resource5batch_normalization_107/FusedBatchNormV3:batch_mean:08^batch_normalization_107/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_107/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_107/AssignNewValue�
(batch_normalization_107/AssignNewValue_1AssignVariableOpBbatch_normalization_107_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_107/FusedBatchNormV3:batch_variance:0:^batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_107/AssignNewValue_1�
conv2d_93/Conv2D/ReadVariableOpReadVariableOp(conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_93/Conv2D/ReadVariableOp�
conv2d_93/Conv2DConv2D,batch_normalization_107/FusedBatchNormV3:y:0'conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_93/Conv2D�
 conv2d_93/BiasAdd/ReadVariableOpReadVariableOp)conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_93/BiasAdd/ReadVariableOp�
conv2d_93/BiasAddBiasAddconv2d_93/Conv2D:output:0(conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_93/BiasAdd~
conv2d_93/ReluReluconv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_93/Relu�
&batch_normalization_108/ReadVariableOpReadVariableOp/batch_normalization_108_readvariableop_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_108/ReadVariableOp�
(batch_normalization_108/ReadVariableOp_1ReadVariableOp1batch_normalization_108_readvariableop_1_resource*
_output_shapes
:@*
dtype02*
(batch_normalization_108/ReadVariableOp_1�
7batch_normalization_108/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_108_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype029
7batch_normalization_108/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_108_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02;
9batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_108/FusedBatchNormV3FusedBatchNormV3conv2d_93/Relu:activations:0.batch_normalization_108/ReadVariableOp:value:00batch_normalization_108/ReadVariableOp_1:value:0?batch_normalization_108/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_108/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_108/FusedBatchNormV3�
&batch_normalization_108/AssignNewValueAssignVariableOp@batch_normalization_108_fusedbatchnormv3_readvariableop_resource5batch_normalization_108/FusedBatchNormV3:batch_mean:08^batch_normalization_108/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_108/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_108/AssignNewValue�
(batch_normalization_108/AssignNewValue_1AssignVariableOpBbatch_normalization_108_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_108/FusedBatchNormV3:batch_variance:0:^batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_108/AssignNewValue_1�
max_pooling2d_46/MaxPoolMaxPool,batch_normalization_108/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_46/MaxPool�
conv2d_94/Conv2D/ReadVariableOpReadVariableOp(conv2d_94_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_94/Conv2D/ReadVariableOp�
conv2d_94/Conv2DConv2D!max_pooling2d_46/MaxPool:output:0'conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_94/Conv2D�
 conv2d_94/BiasAdd/ReadVariableOpReadVariableOp)conv2d_94_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_94/BiasAdd/ReadVariableOp�
conv2d_94/BiasAddBiasAddconv2d_94/Conv2D:output:0(conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_94/BiasAdd
conv2d_94/ReluReluconv2d_94/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_94/Relu�
&batch_normalization_109/ReadVariableOpReadVariableOp/batch_normalization_109_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_109/ReadVariableOp�
(batch_normalization_109/ReadVariableOp_1ReadVariableOp1batch_normalization_109_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_109/ReadVariableOp_1�
7batch_normalization_109/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_109_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_109/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_109_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_109/FusedBatchNormV3FusedBatchNormV3conv2d_94/Relu:activations:0.batch_normalization_109/ReadVariableOp:value:00batch_normalization_109/ReadVariableOp_1:value:0?batch_normalization_109/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_109/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_109/FusedBatchNormV3�
&batch_normalization_109/AssignNewValueAssignVariableOp@batch_normalization_109_fusedbatchnormv3_readvariableop_resource5batch_normalization_109/FusedBatchNormV3:batch_mean:08^batch_normalization_109/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_109/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_109/AssignNewValue�
(batch_normalization_109/AssignNewValue_1AssignVariableOpBbatch_normalization_109_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_109/FusedBatchNormV3:batch_variance:0:^batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_109/AssignNewValue_1�
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_95/Conv2D/ReadVariableOp�
conv2d_95/Conv2DConv2D,batch_normalization_109/FusedBatchNormV3:y:0'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_95/Conv2D�
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp�
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_95/BiasAdd
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_95/Relu�
&batch_normalization_110/ReadVariableOpReadVariableOp/batch_normalization_110_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_110/ReadVariableOp�
(batch_normalization_110/ReadVariableOp_1ReadVariableOp1batch_normalization_110_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_110/ReadVariableOp_1�
7batch_normalization_110/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_110_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_110/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_110_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_110/FusedBatchNormV3FusedBatchNormV3conv2d_95/Relu:activations:0.batch_normalization_110/ReadVariableOp:value:00batch_normalization_110/ReadVariableOp_1:value:0?batch_normalization_110/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_110/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_110/FusedBatchNormV3�
&batch_normalization_110/AssignNewValueAssignVariableOp@batch_normalization_110_fusedbatchnormv3_readvariableop_resource5batch_normalization_110/FusedBatchNormV3:batch_mean:08^batch_normalization_110/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_110/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_110/AssignNewValue�
(batch_normalization_110/AssignNewValue_1AssignVariableOpBbatch_normalization_110_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_110/FusedBatchNormV3:batch_variance:0:^batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_110/AssignNewValue_1�
max_pooling2d_47/MaxPoolMaxPool,batch_normalization_110/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_47/MaxPool�
&batch_normalization_111/ReadVariableOpReadVariableOp/batch_normalization_111_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_111/ReadVariableOp�
(batch_normalization_111/ReadVariableOp_1ReadVariableOp1batch_normalization_111_readvariableop_1_resource*
_output_shapes	
:�*
dtype02*
(batch_normalization_111/ReadVariableOp_1�
7batch_normalization_111/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_111_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_111/FusedBatchNormV3/ReadVariableOp�
9batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_111_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02;
9batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1�
(batch_normalization_111/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_47/MaxPool:output:0.batch_normalization_111/ReadVariableOp:value:00batch_normalization_111/ReadVariableOp_1:value:0?batch_normalization_111/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_111/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2*
(batch_normalization_111/FusedBatchNormV3�
&batch_normalization_111/AssignNewValueAssignVariableOp@batch_normalization_111_fusedbatchnormv3_readvariableop_resource5batch_normalization_111/FusedBatchNormV3:batch_mean:08^batch_normalization_111/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*S
_classI
GEloc:@batch_normalization_111/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02(
&batch_normalization_111/AssignNewValue�
(batch_normalization_111/AssignNewValue_1AssignVariableOpBbatch_normalization_111_fusedbatchnormv3_readvariableop_1_resource9batch_normalization_111/FusedBatchNormV3:batch_variance:0:^batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02*
(batch_normalization_111/AssignNewValue_1u
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_15/Const�
flatten_15/ReshapeReshape,batch_normalization_111/FusedBatchNormV3:y:0flatten_15/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_15/Reshape�
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_16/MatMul/ReadVariableOp�
dense_16/MatMulMatMulflatten_15/Reshape:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_16/MatMul�
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_16/BiasAdd/ReadVariableOp�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_16/BiasAdd}
dense_16/SigmoidSigmoiddense_16/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_16/Sigmoid�
lambda_16/l2_normalize/SquareSquaredense_16/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
lambda_16/l2_normalize/Square�
,lambda_16/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,lambda_16/l2_normalize/Sum/reduction_indices�
lambda_16/l2_normalize/SumSum!lambda_16/l2_normalize/Square:y:05lambda_16/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
lambda_16/l2_normalize/Sum�
 lambda_16/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2"
 lambda_16/l2_normalize/Maximum/y�
lambda_16/l2_normalize/MaximumMaximum#lambda_16/l2_normalize/Sum:output:0)lambda_16/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2 
lambda_16/l2_normalize/Maximum�
lambda_16/l2_normalize/RsqrtRsqrt"lambda_16/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
lambda_16/l2_normalize/Rsqrt�
lambda_16/l2_normalizeMuldense_16/Sigmoid:y:0 lambda_16/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
lambda_16/l2_normalize�
IdentityIdentitylambda_16/l2_normalize:z:0'^batch_normalization_105/AssignNewValue)^batch_normalization_105/AssignNewValue_18^batch_normalization_105/FusedBatchNormV3/ReadVariableOp:^batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_105/ReadVariableOp)^batch_normalization_105/ReadVariableOp_1'^batch_normalization_106/AssignNewValue)^batch_normalization_106/AssignNewValue_18^batch_normalization_106/FusedBatchNormV3/ReadVariableOp:^batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_106/ReadVariableOp)^batch_normalization_106/ReadVariableOp_1'^batch_normalization_107/AssignNewValue)^batch_normalization_107/AssignNewValue_18^batch_normalization_107/FusedBatchNormV3/ReadVariableOp:^batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_107/ReadVariableOp)^batch_normalization_107/ReadVariableOp_1'^batch_normalization_108/AssignNewValue)^batch_normalization_108/AssignNewValue_18^batch_normalization_108/FusedBatchNormV3/ReadVariableOp:^batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_108/ReadVariableOp)^batch_normalization_108/ReadVariableOp_1'^batch_normalization_109/AssignNewValue)^batch_normalization_109/AssignNewValue_18^batch_normalization_109/FusedBatchNormV3/ReadVariableOp:^batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_109/ReadVariableOp)^batch_normalization_109/ReadVariableOp_1'^batch_normalization_110/AssignNewValue)^batch_normalization_110/AssignNewValue_18^batch_normalization_110/FusedBatchNormV3/ReadVariableOp:^batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_110/ReadVariableOp)^batch_normalization_110/ReadVariableOp_1'^batch_normalization_111/AssignNewValue)^batch_normalization_111/AssignNewValue_18^batch_normalization_111/FusedBatchNormV3/ReadVariableOp:^batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_111/ReadVariableOp)^batch_normalization_111/ReadVariableOp_1!^conv2d_90/BiasAdd/ReadVariableOp ^conv2d_90/Conv2D/ReadVariableOp!^conv2d_91/BiasAdd/ReadVariableOp ^conv2d_91/Conv2D/ReadVariableOp!^conv2d_92/BiasAdd/ReadVariableOp ^conv2d_92/Conv2D/ReadVariableOp!^conv2d_93/BiasAdd/ReadVariableOp ^conv2d_93/Conv2D/ReadVariableOp!^conv2d_94/BiasAdd/ReadVariableOp ^conv2d_94/Conv2D/ReadVariableOp!^conv2d_95/BiasAdd/ReadVariableOp ^conv2d_95/Conv2D/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2P
&batch_normalization_105/AssignNewValue&batch_normalization_105/AssignNewValue2T
(batch_normalization_105/AssignNewValue_1(batch_normalization_105/AssignNewValue_12r
7batch_normalization_105/FusedBatchNormV3/ReadVariableOp7batch_normalization_105/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_105/FusedBatchNormV3/ReadVariableOp_19batch_normalization_105/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_105/ReadVariableOp&batch_normalization_105/ReadVariableOp2T
(batch_normalization_105/ReadVariableOp_1(batch_normalization_105/ReadVariableOp_12P
&batch_normalization_106/AssignNewValue&batch_normalization_106/AssignNewValue2T
(batch_normalization_106/AssignNewValue_1(batch_normalization_106/AssignNewValue_12r
7batch_normalization_106/FusedBatchNormV3/ReadVariableOp7batch_normalization_106/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_19batch_normalization_106/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_106/ReadVariableOp&batch_normalization_106/ReadVariableOp2T
(batch_normalization_106/ReadVariableOp_1(batch_normalization_106/ReadVariableOp_12P
&batch_normalization_107/AssignNewValue&batch_normalization_107/AssignNewValue2T
(batch_normalization_107/AssignNewValue_1(batch_normalization_107/AssignNewValue_12r
7batch_normalization_107/FusedBatchNormV3/ReadVariableOp7batch_normalization_107/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_19batch_normalization_107/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_107/ReadVariableOp&batch_normalization_107/ReadVariableOp2T
(batch_normalization_107/ReadVariableOp_1(batch_normalization_107/ReadVariableOp_12P
&batch_normalization_108/AssignNewValue&batch_normalization_108/AssignNewValue2T
(batch_normalization_108/AssignNewValue_1(batch_normalization_108/AssignNewValue_12r
7batch_normalization_108/FusedBatchNormV3/ReadVariableOp7batch_normalization_108/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_108/FusedBatchNormV3/ReadVariableOp_19batch_normalization_108/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_108/ReadVariableOp&batch_normalization_108/ReadVariableOp2T
(batch_normalization_108/ReadVariableOp_1(batch_normalization_108/ReadVariableOp_12P
&batch_normalization_109/AssignNewValue&batch_normalization_109/AssignNewValue2T
(batch_normalization_109/AssignNewValue_1(batch_normalization_109/AssignNewValue_12r
7batch_normalization_109/FusedBatchNormV3/ReadVariableOp7batch_normalization_109/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_109/FusedBatchNormV3/ReadVariableOp_19batch_normalization_109/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_109/ReadVariableOp&batch_normalization_109/ReadVariableOp2T
(batch_normalization_109/ReadVariableOp_1(batch_normalization_109/ReadVariableOp_12P
&batch_normalization_110/AssignNewValue&batch_normalization_110/AssignNewValue2T
(batch_normalization_110/AssignNewValue_1(batch_normalization_110/AssignNewValue_12r
7batch_normalization_110/FusedBatchNormV3/ReadVariableOp7batch_normalization_110/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_110/FusedBatchNormV3/ReadVariableOp_19batch_normalization_110/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_110/ReadVariableOp&batch_normalization_110/ReadVariableOp2T
(batch_normalization_110/ReadVariableOp_1(batch_normalization_110/ReadVariableOp_12P
&batch_normalization_111/AssignNewValue&batch_normalization_111/AssignNewValue2T
(batch_normalization_111/AssignNewValue_1(batch_normalization_111/AssignNewValue_12r
7batch_normalization_111/FusedBatchNormV3/ReadVariableOp7batch_normalization_111/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_111/FusedBatchNormV3/ReadVariableOp_19batch_normalization_111/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_111/ReadVariableOp&batch_normalization_111/ReadVariableOp2T
(batch_normalization_111/ReadVariableOp_1(batch_normalization_111/ReadVariableOp_12D
 conv2d_90/BiasAdd/ReadVariableOp conv2d_90/BiasAdd/ReadVariableOp2B
conv2d_90/Conv2D/ReadVariableOpconv2d_90/Conv2D/ReadVariableOp2D
 conv2d_91/BiasAdd/ReadVariableOp conv2d_91/BiasAdd/ReadVariableOp2B
conv2d_91/Conv2D/ReadVariableOpconv2d_91/Conv2D/ReadVariableOp2D
 conv2d_92/BiasAdd/ReadVariableOp conv2d_92/BiasAdd/ReadVariableOp2B
conv2d_92/Conv2D/ReadVariableOpconv2d_92/Conv2D/ReadVariableOp2D
 conv2d_93/BiasAdd/ReadVariableOp conv2d_93/BiasAdd/ReadVariableOp2B
conv2d_93/Conv2D/ReadVariableOpconv2d_93/Conv2D/ReadVariableOp2D
 conv2d_94/BiasAdd/ReadVariableOp conv2d_94/BiasAdd/ReadVariableOp2B
conv2d_94/Conv2D/ReadVariableOpconv2d_94/Conv2D/ReadVariableOp2D
 conv2d_95/BiasAdd/ReadVariableOp conv2d_95/BiasAdd/ReadVariableOp2B
conv2d_95/Conv2D/ReadVariableOpconv2d_95/Conv2D/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219982

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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220260

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
�

*__inference_conv2d_92_layer_call_fn_220176

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
E__inference_conv2d_92_layer_call_and_return_conditional_losses_2182102
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
�j
�
I__inference_sequential_15_layer_call_and_return_conditional_losses_219164

inputs
conv2d_90_219060
conv2d_90_219062"
batch_normalization_105_219065"
batch_normalization_105_219067"
batch_normalization_105_219069"
batch_normalization_105_219071
conv2d_91_219074
conv2d_91_219076"
batch_normalization_106_219079"
batch_normalization_106_219081"
batch_normalization_106_219083"
batch_normalization_106_219085
conv2d_92_219089
conv2d_92_219091"
batch_normalization_107_219094"
batch_normalization_107_219096"
batch_normalization_107_219098"
batch_normalization_107_219100
conv2d_93_219103
conv2d_93_219105"
batch_normalization_108_219108"
batch_normalization_108_219110"
batch_normalization_108_219112"
batch_normalization_108_219114
conv2d_94_219118
conv2d_94_219120"
batch_normalization_109_219123"
batch_normalization_109_219125"
batch_normalization_109_219127"
batch_normalization_109_219129
conv2d_95_219132
conv2d_95_219134"
batch_normalization_110_219137"
batch_normalization_110_219139"
batch_normalization_110_219141"
batch_normalization_110_219143"
batch_normalization_111_219147"
batch_normalization_111_219149"
batch_normalization_111_219151"
batch_normalization_111_219153
dense_16_219157
dense_16_219159
identity��/batch_normalization_105/StatefulPartitionedCall�/batch_normalization_106/StatefulPartitionedCall�/batch_normalization_107/StatefulPartitionedCall�/batch_normalization_108/StatefulPartitionedCall�/batch_normalization_109/StatefulPartitionedCall�/batch_normalization_110/StatefulPartitionedCall�/batch_normalization_111/StatefulPartitionedCall�!conv2d_90/StatefulPartitionedCall�!conv2d_91/StatefulPartitionedCall�!conv2d_92/StatefulPartitionedCall�!conv2d_93/StatefulPartitionedCall�!conv2d_94/StatefulPartitionedCall�!conv2d_95/StatefulPartitionedCall� dense_16/StatefulPartitionedCall�
!conv2d_90/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_90_219060conv2d_90_219062*
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
E__inference_conv2d_90_layer_call_and_return_conditional_losses_2180092#
!conv2d_90/StatefulPartitionedCall�
/batch_normalization_105/StatefulPartitionedCallStatefulPartitionedCall*conv2d_90/StatefulPartitionedCall:output:0batch_normalization_105_219065batch_normalization_105_219067batch_normalization_105_219069batch_normalization_105_219071*
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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_21806221
/batch_normalization_105/StatefulPartitionedCall�
!conv2d_91/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_105/StatefulPartitionedCall:output:0conv2d_91_219074conv2d_91_219076*
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
E__inference_conv2d_91_layer_call_and_return_conditional_losses_2181092#
!conv2d_91/StatefulPartitionedCall�
/batch_normalization_106/StatefulPartitionedCallStatefulPartitionedCall*conv2d_91/StatefulPartitionedCall:output:0batch_normalization_106_219079batch_normalization_106_219081batch_normalization_106_219083batch_normalization_106_219085*
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
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_21816221
/batch_normalization_106/StatefulPartitionedCall�
 max_pooling2d_45/PartitionedCallPartitionedCall8batch_normalization_106/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_2174442"
 max_pooling2d_45/PartitionedCall�
!conv2d_92/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0conv2d_92_219089conv2d_92_219091*
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
E__inference_conv2d_92_layer_call_and_return_conditional_losses_2182102#
!conv2d_92/StatefulPartitionedCall�
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall*conv2d_92/StatefulPartitionedCall:output:0batch_normalization_107_219094batch_normalization_107_219096batch_normalization_107_219098batch_normalization_107_219100*
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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_21826321
/batch_normalization_107/StatefulPartitionedCall�
!conv2d_93/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0conv2d_93_219103conv2d_93_219105*
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
E__inference_conv2d_93_layer_call_and_return_conditional_losses_2183102#
!conv2d_93/StatefulPartitionedCall�
/batch_normalization_108/StatefulPartitionedCallStatefulPartitionedCall*conv2d_93/StatefulPartitionedCall:output:0batch_normalization_108_219108batch_normalization_108_219110batch_normalization_108_219112batch_normalization_108_219114*
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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_21836321
/batch_normalization_108/StatefulPartitionedCall�
 max_pooling2d_46/PartitionedCallPartitionedCall8batch_normalization_108/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_2176642"
 max_pooling2d_46/PartitionedCall�
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0conv2d_94_219118conv2d_94_219120*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_2184112#
!conv2d_94/StatefulPartitionedCall�
/batch_normalization_109/StatefulPartitionedCallStatefulPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0batch_normalization_109_219123batch_normalization_109_219125batch_normalization_109_219127batch_normalization_109_219129*
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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_21846421
/batch_normalization_109/StatefulPartitionedCall�
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_109/StatefulPartitionedCall:output:0conv2d_95_219132conv2d_95_219134*
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
GPU 2J 8� *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_2185112#
!conv2d_95/StatefulPartitionedCall�
/batch_normalization_110/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_110_219137batch_normalization_110_219139batch_normalization_110_219141batch_normalization_110_219143*
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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_21856421
/batch_normalization_110/StatefulPartitionedCall�
 max_pooling2d_47/PartitionedCallPartitionedCall8batch_normalization_110/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2178842"
 max_pooling2d_47/PartitionedCall�
/batch_normalization_111/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0batch_normalization_111_219147batch_normalization_111_219149batch_normalization_111_219151batch_normalization_111_219153*
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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_21863821
/batch_normalization_111/StatefulPartitionedCall�
flatten_15/PartitionedCallPartitionedCall8batch_normalization_111/StatefulPartitionedCall:output:0*
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
F__inference_flatten_15_layer_call_and_return_conditional_losses_2186802
flatten_15/PartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_16_219157dense_16_219159*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_2186992"
 dense_16/StatefulPartitionedCall�
lambda_16/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
E__inference_lambda_16_layer_call_and_return_conditional_losses_2187372
lambda_16/PartitionedCall�
IdentityIdentity"lambda_16/PartitionedCall:output:00^batch_normalization_105/StatefulPartitionedCall0^batch_normalization_106/StatefulPartitionedCall0^batch_normalization_107/StatefulPartitionedCall0^batch_normalization_108/StatefulPartitionedCall0^batch_normalization_109/StatefulPartitionedCall0^batch_normalization_110/StatefulPartitionedCall0^batch_normalization_111/StatefulPartitionedCall"^conv2d_90/StatefulPartitionedCall"^conv2d_91/StatefulPartitionedCall"^conv2d_92/StatefulPartitionedCall"^conv2d_93/StatefulPartitionedCall"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2b
/batch_normalization_105/StatefulPartitionedCall/batch_normalization_105/StatefulPartitionedCall2b
/batch_normalization_106/StatefulPartitionedCall/batch_normalization_106/StatefulPartitionedCall2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2b
/batch_normalization_108/StatefulPartitionedCall/batch_normalization_108/StatefulPartitionedCall2b
/batch_normalization_109/StatefulPartitionedCall/batch_normalization_109/StatefulPartitionedCall2b
/batch_normalization_110/StatefulPartitionedCall/batch_normalization_110/StatefulPartitionedCall2b
/batch_normalization_111/StatefulPartitionedCall/batch_normalization_111/StatefulPartitionedCall2F
!conv2d_90/StatefulPartitionedCall!conv2d_90/StatefulPartitionedCall2F
!conv2d_91/StatefulPartitionedCall!conv2d_91/StatefulPartitionedCall2F
!conv2d_92/StatefulPartitionedCall!conv2d_92/StatefulPartitionedCall2F
!conv2d_93/StatefulPartitionedCall!conv2d_93/StatefulPartitionedCall2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220130

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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220196

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
�
M
1__inference_max_pooling2d_47_layer_call_fn_217890

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
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_2178842
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
�
�
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220066

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
�
�
8__inference_batch_normalization_107_layer_call_fn_220240

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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_2175432
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
�
�
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_218546

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
8__inference_batch_normalization_105_layer_call_fn_219944

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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_2173232
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
�
�
8__inference_batch_normalization_110_layer_call_fn_220684

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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_2178672
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
8__inference_batch_normalization_111_layer_call_fn_220812

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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_2186382
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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_217732

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
8__inference_batch_normalization_105_layer_call_fn_219931

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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_2172922
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
�

�
E__inference_conv2d_93_layer_call_and_return_conditional_losses_218310

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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220344

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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220786

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
�

*__inference_conv2d_91_layer_call_fn_220028

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
E__inference_conv2d_91_layer_call_and_return_conditional_losses_2181092
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
�
�
8__inference_batch_normalization_108_layer_call_fn_220439

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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_2176162
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
�
�
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220658

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
�
�
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_218363

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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_217512

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

a
E__inference_lambda_16_layer_call_and_return_conditional_losses_220918

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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_218464

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
�[
�
__inference__traced_save_221088
file_prefix/
+savev2_conv2d_90_kernel_read_readvariableop-
)savev2_conv2d_90_bias_read_readvariableop<
8savev2_batch_normalization_105_gamma_read_readvariableop;
7savev2_batch_normalization_105_beta_read_readvariableopB
>savev2_batch_normalization_105_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_105_moving_variance_read_readvariableop/
+savev2_conv2d_91_kernel_read_readvariableop-
)savev2_conv2d_91_bias_read_readvariableop<
8savev2_batch_normalization_106_gamma_read_readvariableop;
7savev2_batch_normalization_106_beta_read_readvariableopB
>savev2_batch_normalization_106_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_106_moving_variance_read_readvariableop/
+savev2_conv2d_92_kernel_read_readvariableop-
)savev2_conv2d_92_bias_read_readvariableop<
8savev2_batch_normalization_107_gamma_read_readvariableop;
7savev2_batch_normalization_107_beta_read_readvariableopB
>savev2_batch_normalization_107_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_107_moving_variance_read_readvariableop/
+savev2_conv2d_93_kernel_read_readvariableop-
)savev2_conv2d_93_bias_read_readvariableop<
8savev2_batch_normalization_108_gamma_read_readvariableop;
7savev2_batch_normalization_108_beta_read_readvariableopB
>savev2_batch_normalization_108_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_108_moving_variance_read_readvariableop/
+savev2_conv2d_94_kernel_read_readvariableop-
)savev2_conv2d_94_bias_read_readvariableop<
8savev2_batch_normalization_109_gamma_read_readvariableop;
7savev2_batch_normalization_109_beta_read_readvariableopB
>savev2_batch_normalization_109_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_109_moving_variance_read_readvariableop/
+savev2_conv2d_95_kernel_read_readvariableop-
)savev2_conv2d_95_bias_read_readvariableop<
8savev2_batch_normalization_110_gamma_read_readvariableop;
7savev2_batch_normalization_110_beta_read_readvariableopB
>savev2_batch_normalization_110_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_110_moving_variance_read_readvariableop<
8savev2_batch_normalization_111_gamma_read_readvariableop;
7savev2_batch_normalization_111_beta_read_readvariableopB
>savev2_batch_normalization_111_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_111_moving_variance_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_90_kernel_read_readvariableop)savev2_conv2d_90_bias_read_readvariableop8savev2_batch_normalization_105_gamma_read_readvariableop7savev2_batch_normalization_105_beta_read_readvariableop>savev2_batch_normalization_105_moving_mean_read_readvariableopBsavev2_batch_normalization_105_moving_variance_read_readvariableop+savev2_conv2d_91_kernel_read_readvariableop)savev2_conv2d_91_bias_read_readvariableop8savev2_batch_normalization_106_gamma_read_readvariableop7savev2_batch_normalization_106_beta_read_readvariableop>savev2_batch_normalization_106_moving_mean_read_readvariableopBsavev2_batch_normalization_106_moving_variance_read_readvariableop+savev2_conv2d_92_kernel_read_readvariableop)savev2_conv2d_92_bias_read_readvariableop8savev2_batch_normalization_107_gamma_read_readvariableop7savev2_batch_normalization_107_beta_read_readvariableop>savev2_batch_normalization_107_moving_mean_read_readvariableopBsavev2_batch_normalization_107_moving_variance_read_readvariableop+savev2_conv2d_93_kernel_read_readvariableop)savev2_conv2d_93_bias_read_readvariableop8savev2_batch_normalization_108_gamma_read_readvariableop7savev2_batch_normalization_108_beta_read_readvariableop>savev2_batch_normalization_108_moving_mean_read_readvariableopBsavev2_batch_normalization_108_moving_variance_read_readvariableop+savev2_conv2d_94_kernel_read_readvariableop)savev2_conv2d_94_bias_read_readvariableop8savev2_batch_normalization_109_gamma_read_readvariableop7savev2_batch_normalization_109_beta_read_readvariableop>savev2_batch_normalization_109_moving_mean_read_readvariableopBsavev2_batch_normalization_109_moving_variance_read_readvariableop+savev2_conv2d_95_kernel_read_readvariableop)savev2_conv2d_95_bias_read_readvariableop8savev2_batch_normalization_110_gamma_read_readvariableop7savev2_batch_normalization_110_beta_read_readvariableop>savev2_batch_normalization_110_moving_mean_read_readvariableopBsavev2_batch_normalization_110_moving_variance_read_readvariableop8savev2_batch_normalization_111_gamma_read_readvariableop7savev2_batch_normalization_111_beta_read_readvariableop>savev2_batch_normalization_111_moving_mean_read_readvariableopBsavev2_batch_normalization_111_moving_variance_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
8__inference_batch_normalization_108_layer_call_fn_220388

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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_2183632
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
�
�
$__inference_signature_wrapper_219342
conv2d_90_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_90_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_2172302
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
_user_specified_nameconv2d_90_input
�

*__inference_conv2d_95_layer_call_fn_220620

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
GPU 2J 8� *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_2185112
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
�

*__inference_conv2d_94_layer_call_fn_220472

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
GPU 2J 8� *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_2184112
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
��
�
"__inference__traced_restore_221224
file_prefix%
!assignvariableop_conv2d_90_kernel%
!assignvariableop_1_conv2d_90_bias4
0assignvariableop_2_batch_normalization_105_gamma3
/assignvariableop_3_batch_normalization_105_beta:
6assignvariableop_4_batch_normalization_105_moving_mean>
:assignvariableop_5_batch_normalization_105_moving_variance'
#assignvariableop_6_conv2d_91_kernel%
!assignvariableop_7_conv2d_91_bias4
0assignvariableop_8_batch_normalization_106_gamma3
/assignvariableop_9_batch_normalization_106_beta;
7assignvariableop_10_batch_normalization_106_moving_mean?
;assignvariableop_11_batch_normalization_106_moving_variance(
$assignvariableop_12_conv2d_92_kernel&
"assignvariableop_13_conv2d_92_bias5
1assignvariableop_14_batch_normalization_107_gamma4
0assignvariableop_15_batch_normalization_107_beta;
7assignvariableop_16_batch_normalization_107_moving_mean?
;assignvariableop_17_batch_normalization_107_moving_variance(
$assignvariableop_18_conv2d_93_kernel&
"assignvariableop_19_conv2d_93_bias5
1assignvariableop_20_batch_normalization_108_gamma4
0assignvariableop_21_batch_normalization_108_beta;
7assignvariableop_22_batch_normalization_108_moving_mean?
;assignvariableop_23_batch_normalization_108_moving_variance(
$assignvariableop_24_conv2d_94_kernel&
"assignvariableop_25_conv2d_94_bias5
1assignvariableop_26_batch_normalization_109_gamma4
0assignvariableop_27_batch_normalization_109_beta;
7assignvariableop_28_batch_normalization_109_moving_mean?
;assignvariableop_29_batch_normalization_109_moving_variance(
$assignvariableop_30_conv2d_95_kernel&
"assignvariableop_31_conv2d_95_bias5
1assignvariableop_32_batch_normalization_110_gamma4
0assignvariableop_33_batch_normalization_110_beta;
7assignvariableop_34_batch_normalization_110_moving_mean?
;assignvariableop_35_batch_normalization_110_moving_variance5
1assignvariableop_36_batch_normalization_111_gamma4
0assignvariableop_37_batch_normalization_111_beta;
7assignvariableop_38_batch_normalization_111_moving_mean?
;assignvariableop_39_batch_normalization_111_moving_variance'
#assignvariableop_40_dense_16_kernel%
!assignvariableop_41_dense_16_bias
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_90_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_90_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_batch_normalization_105_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_batch_normalization_105_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_105_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp:assignvariableop_5_batch_normalization_105_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_91_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_91_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_batch_normalization_106_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batch_normalization_106_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_106_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp;assignvariableop_11_batch_normalization_106_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_92_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_92_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_batch_normalization_107_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_107_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp7assignvariableop_16_batch_normalization_107_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp;assignvariableop_17_batch_normalization_107_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_93_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_93_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp1assignvariableop_20_batch_normalization_108_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_108_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp7assignvariableop_22_batch_normalization_108_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp;assignvariableop_23_batch_normalization_108_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_94_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_94_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp1assignvariableop_26_batch_normalization_109_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp0assignvariableop_27_batch_normalization_109_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp7assignvariableop_28_batch_normalization_109_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp;assignvariableop_29_batch_normalization_109_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_95_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_95_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp1assignvariableop_32_batch_normalization_110_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp0assignvariableop_33_batch_normalization_110_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp7assignvariableop_34_batch_normalization_110_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp;assignvariableop_35_batch_normalization_110_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp1assignvariableop_36_batch_normalization_111_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp0assignvariableop_37_batch_normalization_111_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp7assignvariableop_38_batch_normalization_111_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp;assignvariableop_39_batch_normalization_111_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp#assignvariableop_40_dense_16_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp!assignvariableop_41_dense_16_biasIdentity_41:output:0"/device:CPU:0*
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
�
�
8__inference_batch_normalization_107_layer_call_fn_220291

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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_2182452
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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_217647

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
E__inference_conv2d_90_layer_call_and_return_conditional_losses_218009

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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_218564

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
M
1__inference_max_pooling2d_46_layer_call_fn_217670

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
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_2176642
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
�
�
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220426

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
�
�
8__inference_batch_normalization_110_layer_call_fn_220735

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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_2185462
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
�
�
8__inference_batch_normalization_105_layer_call_fn_220008

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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_2180622
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
�

*__inference_conv2d_90_layer_call_fn_219880

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
E__inference_conv2d_90_layer_call_and_return_conditional_losses_2180092
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
�
F
*__inference_lambda_16_layer_call_fn_220934

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
E__inference_lambda_16_layer_call_and_return_conditional_losses_2187262
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
�
�
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_217836

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
�
�
8__inference_batch_normalization_108_layer_call_fn_220375

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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_2183452
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
�

�
E__inference_conv2d_92_layer_call_and_return_conditional_losses_220167

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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220722

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
�
�
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_218620

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
�
h
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_217444

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
�
�
8__inference_batch_normalization_110_layer_call_fn_220671

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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_2178362
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
8__inference_batch_normalization_107_layer_call_fn_220304

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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_2182632
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
�
�
.__inference_sequential_15_layer_call_fn_219251
conv2d_90_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_90_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_15_layer_call_and_return_conditional_losses_2191642
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
_user_specified_nameconv2d_90_input
�	
�
D__inference_dense_16_layer_call_and_return_conditional_losses_218699

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
�

�
E__inference_conv2d_95_layer_call_and_return_conditional_losses_220611

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
�
~
)__inference_dense_16_layer_call_fn_220907

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
D__inference_dense_16_layer_call_and_return_conditional_losses_2186992
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
�
�
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_217867

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

�
E__inference_conv2d_94_layer_call_and_return_conditional_losses_220463

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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219964

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
�

�
E__inference_conv2d_90_layer_call_and_return_conditional_losses_219871

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
�
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_218680

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
�
�
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220112

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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220574

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
�
�
8__inference_batch_normalization_106_layer_call_fn_220079

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
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_2173962
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
�
G
+__inference_flatten_15_layer_call_fn_220887

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
F__inference_flatten_15_layer_call_and_return_conditional_losses_2186802
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
�
�
.__inference_sequential_15_layer_call_fn_219055
conv2d_90_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_90_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_15_layer_call_and_return_conditional_losses_2189682
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
_user_specified_nameconv2d_90_input
�
�
8__inference_batch_normalization_109_layer_call_fn_220600

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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_2184642
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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220832

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
�
�
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220510

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

�
E__inference_conv2d_94_layer_call_and_return_conditional_losses_218411

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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_217616

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

a
E__inference_lambda_16_layer_call_and_return_conditional_losses_218737

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
�
�
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_218062

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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_218446

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
�
�
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_217396

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
�
�
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220278

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
�
�
8__inference_batch_normalization_109_layer_call_fn_220523

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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_2177322
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
8__inference_batch_normalization_107_layer_call_fn_220227

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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_2175122
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
�
M
1__inference_max_pooling2d_45_layer_call_fn_217450

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
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_2174442
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
�

�
E__inference_conv2d_93_layer_call_and_return_conditional_losses_220315

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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220768

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
�
�
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_218162

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
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_217427

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
�
F
*__inference_lambda_16_layer_call_fn_220939

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
E__inference_lambda_16_layer_call_and_return_conditional_losses_2187372
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
8__inference_batch_normalization_111_layer_call_fn_220876

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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_2179832
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
8__inference_batch_normalization_105_layer_call_fn_219995

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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_2180442
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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220408

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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220640

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
�
�
8__inference_batch_normalization_111_layer_call_fn_220799

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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_2186202
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
�

�
E__inference_conv2d_91_layer_call_and_return_conditional_losses_220019

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
��
�+
!__inference__wrapped_model_217230
conv2d_90_input:
6sequential_15_conv2d_90_conv2d_readvariableop_resource;
7sequential_15_conv2d_90_biasadd_readvariableop_resourceA
=sequential_15_batch_normalization_105_readvariableop_resourceC
?sequential_15_batch_normalization_105_readvariableop_1_resourceR
Nsequential_15_batch_normalization_105_fusedbatchnormv3_readvariableop_resourceT
Psequential_15_batch_normalization_105_fusedbatchnormv3_readvariableop_1_resource:
6sequential_15_conv2d_91_conv2d_readvariableop_resource;
7sequential_15_conv2d_91_biasadd_readvariableop_resourceA
=sequential_15_batch_normalization_106_readvariableop_resourceC
?sequential_15_batch_normalization_106_readvariableop_1_resourceR
Nsequential_15_batch_normalization_106_fusedbatchnormv3_readvariableop_resourceT
Psequential_15_batch_normalization_106_fusedbatchnormv3_readvariableop_1_resource:
6sequential_15_conv2d_92_conv2d_readvariableop_resource;
7sequential_15_conv2d_92_biasadd_readvariableop_resourceA
=sequential_15_batch_normalization_107_readvariableop_resourceC
?sequential_15_batch_normalization_107_readvariableop_1_resourceR
Nsequential_15_batch_normalization_107_fusedbatchnormv3_readvariableop_resourceT
Psequential_15_batch_normalization_107_fusedbatchnormv3_readvariableop_1_resource:
6sequential_15_conv2d_93_conv2d_readvariableop_resource;
7sequential_15_conv2d_93_biasadd_readvariableop_resourceA
=sequential_15_batch_normalization_108_readvariableop_resourceC
?sequential_15_batch_normalization_108_readvariableop_1_resourceR
Nsequential_15_batch_normalization_108_fusedbatchnormv3_readvariableop_resourceT
Psequential_15_batch_normalization_108_fusedbatchnormv3_readvariableop_1_resource:
6sequential_15_conv2d_94_conv2d_readvariableop_resource;
7sequential_15_conv2d_94_biasadd_readvariableop_resourceA
=sequential_15_batch_normalization_109_readvariableop_resourceC
?sequential_15_batch_normalization_109_readvariableop_1_resourceR
Nsequential_15_batch_normalization_109_fusedbatchnormv3_readvariableop_resourceT
Psequential_15_batch_normalization_109_fusedbatchnormv3_readvariableop_1_resource:
6sequential_15_conv2d_95_conv2d_readvariableop_resource;
7sequential_15_conv2d_95_biasadd_readvariableop_resourceA
=sequential_15_batch_normalization_110_readvariableop_resourceC
?sequential_15_batch_normalization_110_readvariableop_1_resourceR
Nsequential_15_batch_normalization_110_fusedbatchnormv3_readvariableop_resourceT
Psequential_15_batch_normalization_110_fusedbatchnormv3_readvariableop_1_resourceA
=sequential_15_batch_normalization_111_readvariableop_resourceC
?sequential_15_batch_normalization_111_readvariableop_1_resourceR
Nsequential_15_batch_normalization_111_fusedbatchnormv3_readvariableop_resourceT
Psequential_15_batch_normalization_111_fusedbatchnormv3_readvariableop_1_resource9
5sequential_15_dense_16_matmul_readvariableop_resource:
6sequential_15_dense_16_biasadd_readvariableop_resource
identity��Esequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp�Gsequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1�4sequential_15/batch_normalization_105/ReadVariableOp�6sequential_15/batch_normalization_105/ReadVariableOp_1�Esequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp�Gsequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1�4sequential_15/batch_normalization_106/ReadVariableOp�6sequential_15/batch_normalization_106/ReadVariableOp_1�Esequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp�Gsequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1�4sequential_15/batch_normalization_107/ReadVariableOp�6sequential_15/batch_normalization_107/ReadVariableOp_1�Esequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp�Gsequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1�4sequential_15/batch_normalization_108/ReadVariableOp�6sequential_15/batch_normalization_108/ReadVariableOp_1�Esequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp�Gsequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1�4sequential_15/batch_normalization_109/ReadVariableOp�6sequential_15/batch_normalization_109/ReadVariableOp_1�Esequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp�Gsequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1�4sequential_15/batch_normalization_110/ReadVariableOp�6sequential_15/batch_normalization_110/ReadVariableOp_1�Esequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp�Gsequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1�4sequential_15/batch_normalization_111/ReadVariableOp�6sequential_15/batch_normalization_111/ReadVariableOp_1�.sequential_15/conv2d_90/BiasAdd/ReadVariableOp�-sequential_15/conv2d_90/Conv2D/ReadVariableOp�.sequential_15/conv2d_91/BiasAdd/ReadVariableOp�-sequential_15/conv2d_91/Conv2D/ReadVariableOp�.sequential_15/conv2d_92/BiasAdd/ReadVariableOp�-sequential_15/conv2d_92/Conv2D/ReadVariableOp�.sequential_15/conv2d_93/BiasAdd/ReadVariableOp�-sequential_15/conv2d_93/Conv2D/ReadVariableOp�.sequential_15/conv2d_94/BiasAdd/ReadVariableOp�-sequential_15/conv2d_94/Conv2D/ReadVariableOp�.sequential_15/conv2d_95/BiasAdd/ReadVariableOp�-sequential_15/conv2d_95/Conv2D/ReadVariableOp�-sequential_15/dense_16/BiasAdd/ReadVariableOp�,sequential_15/dense_16/MatMul/ReadVariableOp�
-sequential_15/conv2d_90/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_15/conv2d_90/Conv2D/ReadVariableOp�
sequential_15/conv2d_90/Conv2DConv2Dconv2d_90_input5sequential_15/conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2 
sequential_15/conv2d_90/Conv2D�
.sequential_15/conv2d_90/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/conv2d_90/BiasAdd/ReadVariableOp�
sequential_15/conv2d_90/BiasAddBiasAdd'sequential_15/conv2d_90/Conv2D:output:06sequential_15/conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2!
sequential_15/conv2d_90/BiasAdd�
sequential_15/conv2d_90/ReluRelu(sequential_15/conv2d_90/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
sequential_15/conv2d_90/Relu�
4sequential_15/batch_normalization_105/ReadVariableOpReadVariableOp=sequential_15_batch_normalization_105_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_15/batch_normalization_105/ReadVariableOp�
6sequential_15/batch_normalization_105/ReadVariableOp_1ReadVariableOp?sequential_15_batch_normalization_105_readvariableop_1_resource*
_output_shapes
: *
dtype028
6sequential_15/batch_normalization_105/ReadVariableOp_1�
Esequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_15_batch_normalization_105_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02G
Esequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp�
Gsequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_15_batch_normalization_105_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gsequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1�
6sequential_15/batch_normalization_105/FusedBatchNormV3FusedBatchNormV3*sequential_15/conv2d_90/Relu:activations:0<sequential_15/batch_normalization_105/ReadVariableOp:value:0>sequential_15/batch_normalization_105/ReadVariableOp_1:value:0Msequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp:value:0Osequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 28
6sequential_15/batch_normalization_105/FusedBatchNormV3�
-sequential_15/conv2d_91/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_91_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02/
-sequential_15/conv2d_91/Conv2D/ReadVariableOp�
sequential_15/conv2d_91/Conv2DConv2D:sequential_15/batch_normalization_105/FusedBatchNormV3:y:05sequential_15/conv2d_91/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2 
sequential_15/conv2d_91/Conv2D�
.sequential_15/conv2d_91/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_91_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/conv2d_91/BiasAdd/ReadVariableOp�
sequential_15/conv2d_91/BiasAddBiasAdd'sequential_15/conv2d_91/Conv2D:output:06sequential_15/conv2d_91/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2!
sequential_15/conv2d_91/BiasAdd�
sequential_15/conv2d_91/ReluRelu(sequential_15/conv2d_91/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
sequential_15/conv2d_91/Relu�
4sequential_15/batch_normalization_106/ReadVariableOpReadVariableOp=sequential_15_batch_normalization_106_readvariableop_resource*
_output_shapes
: *
dtype026
4sequential_15/batch_normalization_106/ReadVariableOp�
6sequential_15/batch_normalization_106/ReadVariableOp_1ReadVariableOp?sequential_15_batch_normalization_106_readvariableop_1_resource*
_output_shapes
: *
dtype028
6sequential_15/batch_normalization_106/ReadVariableOp_1�
Esequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_15_batch_normalization_106_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02G
Esequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp�
Gsequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_15_batch_normalization_106_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gsequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1�
6sequential_15/batch_normalization_106/FusedBatchNormV3FusedBatchNormV3*sequential_15/conv2d_91/Relu:activations:0<sequential_15/batch_normalization_106/ReadVariableOp:value:0>sequential_15/batch_normalization_106/ReadVariableOp_1:value:0Msequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp:value:0Osequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 28
6sequential_15/batch_normalization_106/FusedBatchNormV3�
&sequential_15/max_pooling2d_45/MaxPoolMaxPool:sequential_15/batch_normalization_106/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2(
&sequential_15/max_pooling2d_45/MaxPool�
-sequential_15/conv2d_92/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_15/conv2d_92/Conv2D/ReadVariableOp�
sequential_15/conv2d_92/Conv2DConv2D/sequential_15/max_pooling2d_45/MaxPool:output:05sequential_15/conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2 
sequential_15/conv2d_92/Conv2D�
.sequential_15/conv2d_92/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_15/conv2d_92/BiasAdd/ReadVariableOp�
sequential_15/conv2d_92/BiasAddBiasAdd'sequential_15/conv2d_92/Conv2D:output:06sequential_15/conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2!
sequential_15/conv2d_92/BiasAdd�
sequential_15/conv2d_92/ReluRelu(sequential_15/conv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential_15/conv2d_92/Relu�
4sequential_15/batch_normalization_107/ReadVariableOpReadVariableOp=sequential_15_batch_normalization_107_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential_15/batch_normalization_107/ReadVariableOp�
6sequential_15/batch_normalization_107/ReadVariableOp_1ReadVariableOp?sequential_15_batch_normalization_107_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6sequential_15/batch_normalization_107/ReadVariableOp_1�
Esequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_15_batch_normalization_107_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02G
Esequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp�
Gsequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_15_batch_normalization_107_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02I
Gsequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1�
6sequential_15/batch_normalization_107/FusedBatchNormV3FusedBatchNormV3*sequential_15/conv2d_92/Relu:activations:0<sequential_15/batch_normalization_107/ReadVariableOp:value:0>sequential_15/batch_normalization_107/ReadVariableOp_1:value:0Msequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp:value:0Osequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 28
6sequential_15/batch_normalization_107/FusedBatchNormV3�
-sequential_15/conv2d_93/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_15/conv2d_93/Conv2D/ReadVariableOp�
sequential_15/conv2d_93/Conv2DConv2D:sequential_15/batch_normalization_107/FusedBatchNormV3:y:05sequential_15/conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2 
sequential_15/conv2d_93/Conv2D�
.sequential_15/conv2d_93/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_15/conv2d_93/BiasAdd/ReadVariableOp�
sequential_15/conv2d_93/BiasAddBiasAdd'sequential_15/conv2d_93/Conv2D:output:06sequential_15/conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2!
sequential_15/conv2d_93/BiasAdd�
sequential_15/conv2d_93/ReluRelu(sequential_15/conv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential_15/conv2d_93/Relu�
4sequential_15/batch_normalization_108/ReadVariableOpReadVariableOp=sequential_15_batch_normalization_108_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential_15/batch_normalization_108/ReadVariableOp�
6sequential_15/batch_normalization_108/ReadVariableOp_1ReadVariableOp?sequential_15_batch_normalization_108_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6sequential_15/batch_normalization_108/ReadVariableOp_1�
Esequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_15_batch_normalization_108_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02G
Esequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp�
Gsequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_15_batch_normalization_108_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02I
Gsequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1�
6sequential_15/batch_normalization_108/FusedBatchNormV3FusedBatchNormV3*sequential_15/conv2d_93/Relu:activations:0<sequential_15/batch_normalization_108/ReadVariableOp:value:0>sequential_15/batch_normalization_108/ReadVariableOp_1:value:0Msequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp:value:0Osequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 28
6sequential_15/batch_normalization_108/FusedBatchNormV3�
&sequential_15/max_pooling2d_46/MaxPoolMaxPool:sequential_15/batch_normalization_108/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2(
&sequential_15/max_pooling2d_46/MaxPool�
-sequential_15/conv2d_94/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_94_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02/
-sequential_15/conv2d_94/Conv2D/ReadVariableOp�
sequential_15/conv2d_94/Conv2DConv2D/sequential_15/max_pooling2d_46/MaxPool:output:05sequential_15/conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2 
sequential_15/conv2d_94/Conv2D�
.sequential_15/conv2d_94/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_94_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.sequential_15/conv2d_94/BiasAdd/ReadVariableOp�
sequential_15/conv2d_94/BiasAddBiasAdd'sequential_15/conv2d_94/Conv2D:output:06sequential_15/conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2!
sequential_15/conv2d_94/BiasAdd�
sequential_15/conv2d_94/ReluRelu(sequential_15/conv2d_94/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_15/conv2d_94/Relu�
4sequential_15/batch_normalization_109/ReadVariableOpReadVariableOp=sequential_15_batch_normalization_109_readvariableop_resource*
_output_shapes	
:�*
dtype026
4sequential_15/batch_normalization_109/ReadVariableOp�
6sequential_15/batch_normalization_109/ReadVariableOp_1ReadVariableOp?sequential_15_batch_normalization_109_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6sequential_15/batch_normalization_109/ReadVariableOp_1�
Esequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_15_batch_normalization_109_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02G
Esequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp�
Gsequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_15_batch_normalization_109_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02I
Gsequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1�
6sequential_15/batch_normalization_109/FusedBatchNormV3FusedBatchNormV3*sequential_15/conv2d_94/Relu:activations:0<sequential_15/batch_normalization_109/ReadVariableOp:value:0>sequential_15/batch_normalization_109/ReadVariableOp_1:value:0Msequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp:value:0Osequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 28
6sequential_15/batch_normalization_109/FusedBatchNormV3�
-sequential_15/conv2d_95/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_95_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02/
-sequential_15/conv2d_95/Conv2D/ReadVariableOp�
sequential_15/conv2d_95/Conv2DConv2D:sequential_15/batch_normalization_109/FusedBatchNormV3:y:05sequential_15/conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2 
sequential_15/conv2d_95/Conv2D�
.sequential_15/conv2d_95/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_95_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.sequential_15/conv2d_95/BiasAdd/ReadVariableOp�
sequential_15/conv2d_95/BiasAddBiasAdd'sequential_15/conv2d_95/Conv2D:output:06sequential_15/conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2!
sequential_15/conv2d_95/BiasAdd�
sequential_15/conv2d_95/ReluRelu(sequential_15/conv2d_95/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_15/conv2d_95/Relu�
4sequential_15/batch_normalization_110/ReadVariableOpReadVariableOp=sequential_15_batch_normalization_110_readvariableop_resource*
_output_shapes	
:�*
dtype026
4sequential_15/batch_normalization_110/ReadVariableOp�
6sequential_15/batch_normalization_110/ReadVariableOp_1ReadVariableOp?sequential_15_batch_normalization_110_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6sequential_15/batch_normalization_110/ReadVariableOp_1�
Esequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_15_batch_normalization_110_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02G
Esequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp�
Gsequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_15_batch_normalization_110_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02I
Gsequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1�
6sequential_15/batch_normalization_110/FusedBatchNormV3FusedBatchNormV3*sequential_15/conv2d_95/Relu:activations:0<sequential_15/batch_normalization_110/ReadVariableOp:value:0>sequential_15/batch_normalization_110/ReadVariableOp_1:value:0Msequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp:value:0Osequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 28
6sequential_15/batch_normalization_110/FusedBatchNormV3�
&sequential_15/max_pooling2d_47/MaxPoolMaxPool:sequential_15/batch_normalization_110/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2(
&sequential_15/max_pooling2d_47/MaxPool�
4sequential_15/batch_normalization_111/ReadVariableOpReadVariableOp=sequential_15_batch_normalization_111_readvariableop_resource*
_output_shapes	
:�*
dtype026
4sequential_15/batch_normalization_111/ReadVariableOp�
6sequential_15/batch_normalization_111/ReadVariableOp_1ReadVariableOp?sequential_15_batch_normalization_111_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6sequential_15/batch_normalization_111/ReadVariableOp_1�
Esequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_15_batch_normalization_111_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02G
Esequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp�
Gsequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_15_batch_normalization_111_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02I
Gsequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1�
6sequential_15/batch_normalization_111/FusedBatchNormV3FusedBatchNormV3/sequential_15/max_pooling2d_47/MaxPool:output:0<sequential_15/batch_normalization_111/ReadVariableOp:value:0>sequential_15/batch_normalization_111/ReadVariableOp_1:value:0Msequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp:value:0Osequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 28
6sequential_15/batch_normalization_111/FusedBatchNormV3�
sequential_15/flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2 
sequential_15/flatten_15/Const�
 sequential_15/flatten_15/ReshapeReshape:sequential_15/batch_normalization_111/FusedBatchNormV3:y:0'sequential_15/flatten_15/Const:output:0*
T0*(
_output_shapes
:����������2"
 sequential_15/flatten_15/Reshape�
,sequential_15/dense_16/MatMul/ReadVariableOpReadVariableOp5sequential_15_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,sequential_15/dense_16/MatMul/ReadVariableOp�
sequential_15/dense_16/MatMulMatMul)sequential_15/flatten_15/Reshape:output:04sequential_15/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_15/dense_16/MatMul�
-sequential_15/dense_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_15_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_15/dense_16/BiasAdd/ReadVariableOp�
sequential_15/dense_16/BiasAddBiasAdd'sequential_15/dense_16/MatMul:product:05sequential_15/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_15/dense_16/BiasAdd�
sequential_15/dense_16/SigmoidSigmoid'sequential_15/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:����������2 
sequential_15/dense_16/Sigmoid�
+sequential_15/lambda_16/l2_normalize/SquareSquare"sequential_15/dense_16/Sigmoid:y:0*
T0*(
_output_shapes
:����������2-
+sequential_15/lambda_16/l2_normalize/Square�
:sequential_15/lambda_16/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_15/lambda_16/l2_normalize/Sum/reduction_indices�
(sequential_15/lambda_16/l2_normalize/SumSum/sequential_15/lambda_16/l2_normalize/Square:y:0Csequential_15/lambda_16/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2*
(sequential_15/lambda_16/l2_normalize/Sum�
.sequential_15/lambda_16/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+20
.sequential_15/lambda_16/l2_normalize/Maximum/y�
,sequential_15/lambda_16/l2_normalize/MaximumMaximum1sequential_15/lambda_16/l2_normalize/Sum:output:07sequential_15/lambda_16/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2.
,sequential_15/lambda_16/l2_normalize/Maximum�
*sequential_15/lambda_16/l2_normalize/RsqrtRsqrt0sequential_15/lambda_16/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2,
*sequential_15/lambda_16/l2_normalize/Rsqrt�
$sequential_15/lambda_16/l2_normalizeMul"sequential_15/dense_16/Sigmoid:y:0.sequential_15/lambda_16/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2&
$sequential_15/lambda_16/l2_normalize�
IdentityIdentity(sequential_15/lambda_16/l2_normalize:z:0F^sequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOpH^sequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp_15^sequential_15/batch_normalization_105/ReadVariableOp7^sequential_15/batch_normalization_105/ReadVariableOp_1F^sequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOpH^sequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_15^sequential_15/batch_normalization_106/ReadVariableOp7^sequential_15/batch_normalization_106/ReadVariableOp_1F^sequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOpH^sequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_15^sequential_15/batch_normalization_107/ReadVariableOp7^sequential_15/batch_normalization_107/ReadVariableOp_1F^sequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOpH^sequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp_15^sequential_15/batch_normalization_108/ReadVariableOp7^sequential_15/batch_normalization_108/ReadVariableOp_1F^sequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOpH^sequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_15^sequential_15/batch_normalization_109/ReadVariableOp7^sequential_15/batch_normalization_109/ReadVariableOp_1F^sequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOpH^sequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_15^sequential_15/batch_normalization_110/ReadVariableOp7^sequential_15/batch_normalization_110/ReadVariableOp_1F^sequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOpH^sequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_15^sequential_15/batch_normalization_111/ReadVariableOp7^sequential_15/batch_normalization_111/ReadVariableOp_1/^sequential_15/conv2d_90/BiasAdd/ReadVariableOp.^sequential_15/conv2d_90/Conv2D/ReadVariableOp/^sequential_15/conv2d_91/BiasAdd/ReadVariableOp.^sequential_15/conv2d_91/Conv2D/ReadVariableOp/^sequential_15/conv2d_92/BiasAdd/ReadVariableOp.^sequential_15/conv2d_92/Conv2D/ReadVariableOp/^sequential_15/conv2d_93/BiasAdd/ReadVariableOp.^sequential_15/conv2d_93/Conv2D/ReadVariableOp/^sequential_15/conv2d_94/BiasAdd/ReadVariableOp.^sequential_15/conv2d_94/Conv2D/ReadVariableOp/^sequential_15/conv2d_95/BiasAdd/ReadVariableOp.^sequential_15/conv2d_95/Conv2D/ReadVariableOp.^sequential_15/dense_16/BiasAdd/ReadVariableOp-^sequential_15/dense_16/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2�
Esequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOpEsequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp2�
Gsequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp_1Gsequential_15/batch_normalization_105/FusedBatchNormV3/ReadVariableOp_12l
4sequential_15/batch_normalization_105/ReadVariableOp4sequential_15/batch_normalization_105/ReadVariableOp2p
6sequential_15/batch_normalization_105/ReadVariableOp_16sequential_15/batch_normalization_105/ReadVariableOp_12�
Esequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOpEsequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp2�
Gsequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1Gsequential_15/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_12l
4sequential_15/batch_normalization_106/ReadVariableOp4sequential_15/batch_normalization_106/ReadVariableOp2p
6sequential_15/batch_normalization_106/ReadVariableOp_16sequential_15/batch_normalization_106/ReadVariableOp_12�
Esequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOpEsequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp2�
Gsequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1Gsequential_15/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_12l
4sequential_15/batch_normalization_107/ReadVariableOp4sequential_15/batch_normalization_107/ReadVariableOp2p
6sequential_15/batch_normalization_107/ReadVariableOp_16sequential_15/batch_normalization_107/ReadVariableOp_12�
Esequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOpEsequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp2�
Gsequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp_1Gsequential_15/batch_normalization_108/FusedBatchNormV3/ReadVariableOp_12l
4sequential_15/batch_normalization_108/ReadVariableOp4sequential_15/batch_normalization_108/ReadVariableOp2p
6sequential_15/batch_normalization_108/ReadVariableOp_16sequential_15/batch_normalization_108/ReadVariableOp_12�
Esequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOpEsequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp2�
Gsequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_1Gsequential_15/batch_normalization_109/FusedBatchNormV3/ReadVariableOp_12l
4sequential_15/batch_normalization_109/ReadVariableOp4sequential_15/batch_normalization_109/ReadVariableOp2p
6sequential_15/batch_normalization_109/ReadVariableOp_16sequential_15/batch_normalization_109/ReadVariableOp_12�
Esequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOpEsequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp2�
Gsequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1Gsequential_15/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_12l
4sequential_15/batch_normalization_110/ReadVariableOp4sequential_15/batch_normalization_110/ReadVariableOp2p
6sequential_15/batch_normalization_110/ReadVariableOp_16sequential_15/batch_normalization_110/ReadVariableOp_12�
Esequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOpEsequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp2�
Gsequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1Gsequential_15/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_12l
4sequential_15/batch_normalization_111/ReadVariableOp4sequential_15/batch_normalization_111/ReadVariableOp2p
6sequential_15/batch_normalization_111/ReadVariableOp_16sequential_15/batch_normalization_111/ReadVariableOp_12`
.sequential_15/conv2d_90/BiasAdd/ReadVariableOp.sequential_15/conv2d_90/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_90/Conv2D/ReadVariableOp-sequential_15/conv2d_90/Conv2D/ReadVariableOp2`
.sequential_15/conv2d_91/BiasAdd/ReadVariableOp.sequential_15/conv2d_91/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_91/Conv2D/ReadVariableOp-sequential_15/conv2d_91/Conv2D/ReadVariableOp2`
.sequential_15/conv2d_92/BiasAdd/ReadVariableOp.sequential_15/conv2d_92/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_92/Conv2D/ReadVariableOp-sequential_15/conv2d_92/Conv2D/ReadVariableOp2`
.sequential_15/conv2d_93/BiasAdd/ReadVariableOp.sequential_15/conv2d_93/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_93/Conv2D/ReadVariableOp-sequential_15/conv2d_93/Conv2D/ReadVariableOp2`
.sequential_15/conv2d_94/BiasAdd/ReadVariableOp.sequential_15/conv2d_94/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_94/Conv2D/ReadVariableOp-sequential_15/conv2d_94/Conv2D/ReadVariableOp2`
.sequential_15/conv2d_95/BiasAdd/ReadVariableOp.sequential_15/conv2d_95/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_95/Conv2D/ReadVariableOp-sequential_15/conv2d_95/Conv2D/ReadVariableOp2^
-sequential_15/dense_16/BiasAdd/ReadVariableOp-sequential_15/dense_16/BiasAdd/ReadVariableOp2\
,sequential_15/dense_16/MatMul/ReadVariableOp,sequential_15/dense_16/MatMul/ReadVariableOp:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_90_input
�
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_220882

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
�
�
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220048

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
�
�
.__inference_sequential_15_layer_call_fn_219860

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
I__inference_sequential_15_layer_call_and_return_conditional_losses_2191642
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
�
�
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_217763

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
*__inference_conv2d_93_layer_call_fn_220324

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
E__inference_conv2d_93_layer_call_and_return_conditional_losses_2183102
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
�
�
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220362

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
conv2d_90_input@
!serving_default_conv2d_90_input:0���������  >
	lambda_161
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
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_90_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_90", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_105", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_91", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_106", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_92", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_107", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_93", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_108", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_94", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_109", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_110", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_47", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_111", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Lambda", "config": {"name": "lambda_16", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPof\nPGlweXRob24taW5wdXQtODEtZDgwNjgwM2NmNWE2PtoIPGxhbWJkYT4YAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_90_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_90", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_105", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_91", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_106", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_92", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_107", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_93", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_108", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_94", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_109", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_110", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_47", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_111", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Lambda", "config": {"name": "lambda_16", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPof\nPGlweXRob24taW5wdXQtODEtZDgwNjgwM2NmNWE2PtoIPGxhbWJkYT4YAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}}}
�


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_90", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_90", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_105", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_105", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�	

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_91", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_91", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_106", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_106", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_92", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_92", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_107", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_107", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�	

Jkernel
Kbias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_93", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_93", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_108", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_108", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

]kernel
^bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_94", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_94", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_109", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_109", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�	

lkernel
mbias
nregularization_losses
otrainable_variables
p	variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_95", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_110", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_110", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�
{regularization_losses
|trainable_variables
}	variables
~	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_47", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_111", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_111", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�kernel
	�bias
�regularization_losses
�trainable_variables
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
�
�regularization_losses
�trainable_variables
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "lambda_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_16", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPof\nPGlweXRob24taW5wdXQtODEtZDgwNjgwM2NmNWE2PtoIPGxhbWJkYT4YAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
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
*:( 2conv2d_90/kernel
: 2conv2d_90/bias
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
+:) 2batch_normalization_105/gamma
*:( 2batch_normalization_105/beta
3:1  (2#batch_normalization_105/moving_mean
7:5  (2'batch_normalization_105/moving_variance
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
*:(  2conv2d_91/kernel
: 2conv2d_91/bias
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
+:) 2batch_normalization_106/gamma
*:( 2batch_normalization_106/beta
3:1  (2#batch_normalization_106/moving_mean
7:5  (2'batch_normalization_106/moving_variance
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
*:( @2conv2d_92/kernel
:@2conv2d_92/bias
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
+:)@2batch_normalization_107/gamma
*:(@2batch_normalization_107/beta
3:1@ (2#batch_normalization_107/moving_mean
7:5@ (2'batch_normalization_107/moving_variance
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
*:(@@2conv2d_93/kernel
:@2conv2d_93/bias
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
+:)@2batch_normalization_108/gamma
*:(@2batch_normalization_108/beta
3:1@ (2#batch_normalization_108/moving_mean
7:5@ (2'batch_normalization_108/moving_variance
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
+:)@�2conv2d_94/kernel
:�2conv2d_94/bias
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
,:*�2batch_normalization_109/gamma
+:)�2batch_normalization_109/beta
4:2� (2#batch_normalization_109/moving_mean
8:6� (2'batch_normalization_109/moving_variance
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
,:*��2conv2d_95/kernel
:�2conv2d_95/bias
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
,:*�2batch_normalization_110/gamma
+:)�2batch_normalization_110/beta
4:2� (2#batch_normalization_110/moving_mean
8:6� (2'batch_normalization_110/moving_variance
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
,:*�2batch_normalization_111/gamma
+:)�2batch_normalization_111/beta
4:2� (2#batch_normalization_111/moving_mean
8:6� (2'batch_normalization_111/moving_variance
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
��2dense_16/kernel
:�2dense_16/bias
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
I__inference_sequential_15_layer_call_and_return_conditional_losses_219519
I__inference_sequential_15_layer_call_and_return_conditional_losses_218858
I__inference_sequential_15_layer_call_and_return_conditional_losses_219682
I__inference_sequential_15_layer_call_and_return_conditional_losses_218751�
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
!__inference__wrapped_model_217230�
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
conv2d_90_input���������  
�2�
.__inference_sequential_15_layer_call_fn_219771
.__inference_sequential_15_layer_call_fn_219860
.__inference_sequential_15_layer_call_fn_219055
.__inference_sequential_15_layer_call_fn_219251�
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
E__inference_conv2d_90_layer_call_and_return_conditional_losses_219871�
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
*__inference_conv2d_90_layer_call_fn_219880�
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
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219918
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219964
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219982
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219900�
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
8__inference_batch_normalization_105_layer_call_fn_219931
8__inference_batch_normalization_105_layer_call_fn_219944
8__inference_batch_normalization_105_layer_call_fn_219995
8__inference_batch_normalization_105_layer_call_fn_220008�
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
E__inference_conv2d_91_layer_call_and_return_conditional_losses_220019�
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
*__inference_conv2d_91_layer_call_fn_220028�
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
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220066
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220112
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220130
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220048�
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
8__inference_batch_normalization_106_layer_call_fn_220156
8__inference_batch_normalization_106_layer_call_fn_220079
8__inference_batch_normalization_106_layer_call_fn_220092
8__inference_batch_normalization_106_layer_call_fn_220143�
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
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_217444�
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
1__inference_max_pooling2d_45_layer_call_fn_217450�
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
E__inference_conv2d_92_layer_call_and_return_conditional_losses_220167�
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
*__inference_conv2d_92_layer_call_fn_220176�
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
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220214
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220196
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220260
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220278�
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
8__inference_batch_normalization_107_layer_call_fn_220304
8__inference_batch_normalization_107_layer_call_fn_220240
8__inference_batch_normalization_107_layer_call_fn_220291
8__inference_batch_normalization_107_layer_call_fn_220227�
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
E__inference_conv2d_93_layer_call_and_return_conditional_losses_220315�
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
*__inference_conv2d_93_layer_call_fn_220324�
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
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220344
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220408
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220362
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220426�
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
8__inference_batch_normalization_108_layer_call_fn_220375
8__inference_batch_normalization_108_layer_call_fn_220439
8__inference_batch_normalization_108_layer_call_fn_220388
8__inference_batch_normalization_108_layer_call_fn_220452�
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
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_217664�
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
1__inference_max_pooling2d_46_layer_call_fn_217670�
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
E__inference_conv2d_94_layer_call_and_return_conditional_losses_220463�
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
*__inference_conv2d_94_layer_call_fn_220472�
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
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220492
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220556
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220574
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220510�
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
8__inference_batch_normalization_109_layer_call_fn_220600
8__inference_batch_normalization_109_layer_call_fn_220536
8__inference_batch_normalization_109_layer_call_fn_220523
8__inference_batch_normalization_109_layer_call_fn_220587�
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
E__inference_conv2d_95_layer_call_and_return_conditional_losses_220611�
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
*__inference_conv2d_95_layer_call_fn_220620�
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
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220658
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220722
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220640
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220704�
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
8__inference_batch_normalization_110_layer_call_fn_220735
8__inference_batch_normalization_110_layer_call_fn_220671
8__inference_batch_normalization_110_layer_call_fn_220748
8__inference_batch_normalization_110_layer_call_fn_220684�
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
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_217884�
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
1__inference_max_pooling2d_47_layer_call_fn_217890�
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
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220768
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220850
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220832
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220786�
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
8__inference_batch_normalization_111_layer_call_fn_220812
8__inference_batch_normalization_111_layer_call_fn_220863
8__inference_batch_normalization_111_layer_call_fn_220799
8__inference_batch_normalization_111_layer_call_fn_220876�
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
F__inference_flatten_15_layer_call_and_return_conditional_losses_220882�
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
+__inference_flatten_15_layer_call_fn_220887�
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
D__inference_dense_16_layer_call_and_return_conditional_losses_220898�
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
)__inference_dense_16_layer_call_fn_220907�
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
E__inference_lambda_16_layer_call_and_return_conditional_losses_220918
E__inference_lambda_16_layer_call_and_return_conditional_losses_220929�
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
*__inference_lambda_16_layer_call_fn_220939
*__inference_lambda_16_layer_call_fn_220934�
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
$__inference_signature_wrapper_219342conv2d_90_input"�
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
!__inference__wrapped_model_217230�0 !"#()/012;<BCDEJKQRST]^defglmstuv������@�=
6�3
1�.
conv2d_90_input���������  
� "6�3
1
	lambda_16$�!
	lambda_16�����������
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219900� !"#M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219918� !"#M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219964r !"#;�8
1�.
(�%
inputs���������   
p
� "-�*
#� 
0���������   
� �
S__inference_batch_normalization_105_layer_call_and_return_conditional_losses_219982r !"#;�8
1�.
(�%
inputs���������   
p 
� "-�*
#� 
0���������   
� �
8__inference_batch_normalization_105_layer_call_fn_219931� !"#M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
8__inference_batch_normalization_105_layer_call_fn_219944� !"#M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
8__inference_batch_normalization_105_layer_call_fn_219995e !"#;�8
1�.
(�%
inputs���������   
p
� " ����������   �
8__inference_batch_normalization_105_layer_call_fn_220008e !"#;�8
1�.
(�%
inputs���������   
p 
� " ����������   �
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220048�/012M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220066�/012M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220112r/012;�8
1�.
(�%
inputs���������   
p
� "-�*
#� 
0���������   
� �
S__inference_batch_normalization_106_layer_call_and_return_conditional_losses_220130r/012;�8
1�.
(�%
inputs���������   
p 
� "-�*
#� 
0���������   
� �
8__inference_batch_normalization_106_layer_call_fn_220079�/012M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
8__inference_batch_normalization_106_layer_call_fn_220092�/012M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
8__inference_batch_normalization_106_layer_call_fn_220143e/012;�8
1�.
(�%
inputs���������   
p
� " ����������   �
8__inference_batch_normalization_106_layer_call_fn_220156e/012;�8
1�.
(�%
inputs���������   
p 
� " ����������   �
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220196�BCDEM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220214�BCDEM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220260rBCDE;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
S__inference_batch_normalization_107_layer_call_and_return_conditional_losses_220278rBCDE;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
8__inference_batch_normalization_107_layer_call_fn_220227�BCDEM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
8__inference_batch_normalization_107_layer_call_fn_220240�BCDEM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
8__inference_batch_normalization_107_layer_call_fn_220291eBCDE;�8
1�.
(�%
inputs���������@
p
� " ����������@�
8__inference_batch_normalization_107_layer_call_fn_220304eBCDE;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220344rQRST;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220362rQRST;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220408�QRSTM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
S__inference_batch_normalization_108_layer_call_and_return_conditional_losses_220426�QRSTM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
8__inference_batch_normalization_108_layer_call_fn_220375eQRST;�8
1�.
(�%
inputs���������@
p
� " ����������@�
8__inference_batch_normalization_108_layer_call_fn_220388eQRST;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
8__inference_batch_normalization_108_layer_call_fn_220439�QRSTM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
8__inference_batch_normalization_108_layer_call_fn_220452�QRSTM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220492�defgN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220510�defgN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220556tdefg<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
S__inference_batch_normalization_109_layer_call_and_return_conditional_losses_220574tdefg<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
8__inference_batch_normalization_109_layer_call_fn_220523�defgN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_109_layer_call_fn_220536�defgN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
8__inference_batch_normalization_109_layer_call_fn_220587gdefg<�9
2�/
)�&
inputs����������
p
� "!������������
8__inference_batch_normalization_109_layer_call_fn_220600gdefg<�9
2�/
)�&
inputs����������
p 
� "!������������
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220640�stuvN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220658�stuvN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220704tstuv<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_220722tstuv<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
8__inference_batch_normalization_110_layer_call_fn_220671�stuvN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_110_layer_call_fn_220684�stuvN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
8__inference_batch_normalization_110_layer_call_fn_220735gstuv<�9
2�/
)�&
inputs����������
p
� "!������������
8__inference_batch_normalization_110_layer_call_fn_220748gstuv<�9
2�/
)�&
inputs����������
p 
� "!������������
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220768x����<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220786x����<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220832�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_220850�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
8__inference_batch_normalization_111_layer_call_fn_220799k����<�9
2�/
)�&
inputs����������
p
� "!������������
8__inference_batch_normalization_111_layer_call_fn_220812k����<�9
2�/
)�&
inputs����������
p 
� "!������������
8__inference_batch_normalization_111_layer_call_fn_220863�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
8__inference_batch_normalization_111_layer_call_fn_220876�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
E__inference_conv2d_90_layer_call_and_return_conditional_losses_219871l7�4
-�*
(�%
inputs���������  
� "-�*
#� 
0���������   
� �
*__inference_conv2d_90_layer_call_fn_219880_7�4
-�*
(�%
inputs���������  
� " ����������   �
E__inference_conv2d_91_layer_call_and_return_conditional_losses_220019l()7�4
-�*
(�%
inputs���������   
� "-�*
#� 
0���������   
� �
*__inference_conv2d_91_layer_call_fn_220028_()7�4
-�*
(�%
inputs���������   
� " ����������   �
E__inference_conv2d_92_layer_call_and_return_conditional_losses_220167l;<7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
*__inference_conv2d_92_layer_call_fn_220176_;<7�4
-�*
(�%
inputs��������� 
� " ����������@�
E__inference_conv2d_93_layer_call_and_return_conditional_losses_220315lJK7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
*__inference_conv2d_93_layer_call_fn_220324_JK7�4
-�*
(�%
inputs���������@
� " ����������@�
E__inference_conv2d_94_layer_call_and_return_conditional_losses_220463m]^7�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
*__inference_conv2d_94_layer_call_fn_220472`]^7�4
-�*
(�%
inputs���������@
� "!������������
E__inference_conv2d_95_layer_call_and_return_conditional_losses_220611nlm8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
*__inference_conv2d_95_layer_call_fn_220620alm8�5
.�+
)�&
inputs����������
� "!������������
D__inference_dense_16_layer_call_and_return_conditional_losses_220898`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
)__inference_dense_16_layer_call_fn_220907S��0�-
&�#
!�
inputs����������
� "������������
F__inference_flatten_15_layer_call_and_return_conditional_losses_220882b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
+__inference_flatten_15_layer_call_fn_220887U8�5
.�+
)�&
inputs����������
� "������������
E__inference_lambda_16_layer_call_and_return_conditional_losses_220918b8�5
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
E__inference_lambda_16_layer_call_and_return_conditional_losses_220929b8�5
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
*__inference_lambda_16_layer_call_fn_220934U8�5
.�+
!�
inputs����������

 
p
� "������������
*__inference_lambda_16_layer_call_fn_220939U8�5
.�+
!�
inputs����������

 
p 
� "������������
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_217444�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_45_layer_call_fn_217450�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_217664�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_46_layer_call_fn_217670�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_217884�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_47_layer_call_fn_217890�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_15_layer_call_and_return_conditional_losses_218751�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_90_input���������  
p

 
� "&�#
�
0����������
� �
I__inference_sequential_15_layer_call_and_return_conditional_losses_218858�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_90_input���������  
p 

 
� "&�#
�
0����������
� �
I__inference_sequential_15_layer_call_and_return_conditional_losses_219519�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
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
I__inference_sequential_15_layer_call_and_return_conditional_losses_219682�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
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
.__inference_sequential_15_layer_call_fn_219055�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_90_input���������  
p

 
� "������������
.__inference_sequential_15_layer_call_fn_219251�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_90_input���������  
p 

 
� "������������
.__inference_sequential_15_layer_call_fn_219771�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
5�2
(�%
inputs���������  
p

 
� "������������
.__inference_sequential_15_layer_call_fn_219860�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
5�2
(�%
inputs���������  
p 

 
� "������������
$__inference_signature_wrapper_219342�0 !"#()/012;<BCDEJKQRST]^defglmstuv������S�P
� 
I�F
D
conv2d_90_input1�.
conv2d_90_input���������  "6�3
1
	lambda_16$�!
	lambda_16����������