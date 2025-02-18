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
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8֙
�
conv2d_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_66/kernel
}
$conv2d_66/kernel/Read/ReadVariableOpReadVariableOpconv2d_66/kernel*&
_output_shapes
: *
dtype0
t
conv2d_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_66/bias
m
"conv2d_66/bias/Read/ReadVariableOpReadVariableOpconv2d_66/bias*
_output_shapes
: *
dtype0
�
batch_normalization_77/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_77/gamma
�
0batch_normalization_77/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_77/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_77/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_77/beta
�
/batch_normalization_77/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_77/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_77/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_77/moving_mean
�
6batch_normalization_77/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_77/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_77/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_77/moving_variance
�
:batch_normalization_77/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_77/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_67/kernel
}
$conv2d_67/kernel/Read/ReadVariableOpReadVariableOpconv2d_67/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_67/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_67/bias
m
"conv2d_67/bias/Read/ReadVariableOpReadVariableOpconv2d_67/bias*
_output_shapes
: *
dtype0
�
batch_normalization_78/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_78/gamma
�
0batch_normalization_78/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_78/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_78/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_78/beta
�
/batch_normalization_78/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_78/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_78/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_78/moving_mean
�
6batch_normalization_78/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_78/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_78/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_78/moving_variance
�
:batch_normalization_78/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_78/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_68/kernel
}
$conv2d_68/kernel/Read/ReadVariableOpReadVariableOpconv2d_68/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_68/bias
m
"conv2d_68/bias/Read/ReadVariableOpReadVariableOpconv2d_68/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_79/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_79/gamma
�
0batch_normalization_79/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_79/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_79/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_79/beta
�
/batch_normalization_79/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_79/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_79/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_79/moving_mean
�
6batch_normalization_79/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_79/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_79/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_79/moving_variance
�
:batch_normalization_79/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_79/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_69/kernel
}
$conv2d_69/kernel/Read/ReadVariableOpReadVariableOpconv2d_69/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_69/bias
m
"conv2d_69/bias/Read/ReadVariableOpReadVariableOpconv2d_69/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_80/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_80/gamma
�
0batch_normalization_80/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_80/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_80/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_80/beta
�
/batch_normalization_80/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_80/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_80/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_80/moving_mean
�
6batch_normalization_80/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_80/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_80/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_80/moving_variance
�
:batch_normalization_80/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_80/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameconv2d_70/kernel
~
$conv2d_70/kernel/Read/ReadVariableOpReadVariableOpconv2d_70/kernel*'
_output_shapes
:@�*
dtype0
u
conv2d_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_70/bias
n
"conv2d_70/bias/Read/ReadVariableOpReadVariableOpconv2d_70/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_81/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_81/gamma
�
0batch_normalization_81/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_81/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_81/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_81/beta
�
/batch_normalization_81/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_81/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_81/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_81/moving_mean
�
6batch_normalization_81/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_81/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_81/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_81/moving_variance
�
:batch_normalization_81/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_81/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_71/kernel

$conv2d_71/kernel/Read/ReadVariableOpReadVariableOpconv2d_71/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_71/bias
n
"conv2d_71/bias/Read/ReadVariableOpReadVariableOpconv2d_71/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_82/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_82/gamma
�
0batch_normalization_82/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_82/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_82/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_82/beta
�
/batch_normalization_82/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_82/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_82/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_82/moving_mean
�
6batch_normalization_82/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_82/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_82/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_82/moving_variance
�
:batch_normalization_82/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_82/moving_variance*
_output_shapes	
:�*
dtype0
�
batch_normalization_83/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_83/gamma
�
0batch_normalization_83/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_83/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_83/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_83/beta
�
/batch_normalization_83/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_83/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_83/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_83/moving_mean
�
6batch_normalization_83/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_83/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_83/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_83/moving_variance
�
:batch_normalization_83/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_83/moving_variance*
_output_shapes	
:�*
dtype0
|
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_11/kernel
u
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel* 
_output_shapes
:
��*
dtype0
s
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_11/bias
l
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
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
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
�
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
�
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
�
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
�
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
R
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
h

]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
�
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
h

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
�
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
R
{regularization_losses
|	variables
}trainable_variables
~	keras_api
�
axis

�gamma
	�beta
�moving_mean
�moving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
 
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
�
regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
	variables
�metrics
�layer_metrics
trainable_variables
 
\Z
VARIABLE_VALUEconv2d_66/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_66/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
	variables
�metrics
�layer_metrics
trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_77/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_77/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_77/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_77/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1
"2
#3

 0
!1
�
$regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
%	variables
�metrics
�layer_metrics
&trainable_variables
\Z
VARIABLE_VALUEconv2d_67/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_67/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
�
*regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
+	variables
�metrics
�layer_metrics
,trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_78/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_78/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_78/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_78/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01
12
23

/0
01
�
3regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
4	variables
�metrics
�layer_metrics
5trainable_variables
 
 
 
�
7regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
8	variables
�metrics
�layer_metrics
9trainable_variables
\Z
VARIABLE_VALUEconv2d_68/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_68/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
�
=regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
>	variables
�metrics
�layer_metrics
?trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_79/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_79/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_79/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_79/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1
D2
E3

B0
C1
�
Fregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
G	variables
�metrics
�layer_metrics
Htrainable_variables
\Z
VARIABLE_VALUEconv2d_69/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_69/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
�
Lregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
M	variables
�metrics
�layer_metrics
Ntrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_80/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_80/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_80/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_80/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1
S2
T3

Q0
R1
�
Uregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
V	variables
�metrics
�layer_metrics
Wtrainable_variables
 
 
 
�
Yregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
Z	variables
�metrics
�layer_metrics
[trainable_variables
\Z
VARIABLE_VALUEconv2d_70/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_70/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
�
_regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
`	variables
�metrics
�layer_metrics
atrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_81/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_81/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_81/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_81/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1
f2
g3

d0
e1
�
hregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
i	variables
�metrics
�layer_metrics
jtrainable_variables
][
VARIABLE_VALUEconv2d_71/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_71/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

l0
m1
�
nregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
o	variables
�metrics
�layer_metrics
ptrainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_82/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_82/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_82/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_82/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

s0
t1
u2
v3

s0
t1
�
wregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
x	variables
�metrics
�layer_metrics
ytrainable_variables
 
 
 
�
{regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
|	variables
�metrics
�layer_metrics
}trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_83/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_83/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_83/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_83/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
�0
�1
�2
�3

�0
�1
�
�regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
�	variables
�metrics
�layer_metrics
�trainable_variables
 
 
 
�
�regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
�	variables
�metrics
�layer_metrics
�trainable_variables
\Z
VARIABLE_VALUEdense_11/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_11/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
�	variables
�metrics
�layer_metrics
�trainable_variables
 
 
 
�
�regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
�	variables
�metrics
�layer_metrics
�trainable_variables
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
 
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
serving_default_conv2d_66_inputPlaceholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_66_inputconv2d_66/kernelconv2d_66/biasbatch_normalization_77/gammabatch_normalization_77/beta"batch_normalization_77/moving_mean&batch_normalization_77/moving_varianceconv2d_67/kernelconv2d_67/biasbatch_normalization_78/gammabatch_normalization_78/beta"batch_normalization_78/moving_mean&batch_normalization_78/moving_varianceconv2d_68/kernelconv2d_68/biasbatch_normalization_79/gammabatch_normalization_79/beta"batch_normalization_79/moving_mean&batch_normalization_79/moving_varianceconv2d_69/kernelconv2d_69/biasbatch_normalization_80/gammabatch_normalization_80/beta"batch_normalization_80/moving_mean&batch_normalization_80/moving_varianceconv2d_70/kernelconv2d_70/biasbatch_normalization_81/gammabatch_normalization_81/beta"batch_normalization_81/moving_mean&batch_normalization_81/moving_varianceconv2d_71/kernelconv2d_71/biasbatch_normalization_82/gammabatch_normalization_82/beta"batch_normalization_82/moving_mean&batch_normalization_82/moving_variancebatch_normalization_83/gammabatch_normalization_83/beta"batch_normalization_83/moving_mean&batch_normalization_83/moving_variancedense_11/kerneldense_11/bias*6
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
$__inference_signature_wrapper_694495
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_66/kernel/Read/ReadVariableOp"conv2d_66/bias/Read/ReadVariableOp0batch_normalization_77/gamma/Read/ReadVariableOp/batch_normalization_77/beta/Read/ReadVariableOp6batch_normalization_77/moving_mean/Read/ReadVariableOp:batch_normalization_77/moving_variance/Read/ReadVariableOp$conv2d_67/kernel/Read/ReadVariableOp"conv2d_67/bias/Read/ReadVariableOp0batch_normalization_78/gamma/Read/ReadVariableOp/batch_normalization_78/beta/Read/ReadVariableOp6batch_normalization_78/moving_mean/Read/ReadVariableOp:batch_normalization_78/moving_variance/Read/ReadVariableOp$conv2d_68/kernel/Read/ReadVariableOp"conv2d_68/bias/Read/ReadVariableOp0batch_normalization_79/gamma/Read/ReadVariableOp/batch_normalization_79/beta/Read/ReadVariableOp6batch_normalization_79/moving_mean/Read/ReadVariableOp:batch_normalization_79/moving_variance/Read/ReadVariableOp$conv2d_69/kernel/Read/ReadVariableOp"conv2d_69/bias/Read/ReadVariableOp0batch_normalization_80/gamma/Read/ReadVariableOp/batch_normalization_80/beta/Read/ReadVariableOp6batch_normalization_80/moving_mean/Read/ReadVariableOp:batch_normalization_80/moving_variance/Read/ReadVariableOp$conv2d_70/kernel/Read/ReadVariableOp"conv2d_70/bias/Read/ReadVariableOp0batch_normalization_81/gamma/Read/ReadVariableOp/batch_normalization_81/beta/Read/ReadVariableOp6batch_normalization_81/moving_mean/Read/ReadVariableOp:batch_normalization_81/moving_variance/Read/ReadVariableOp$conv2d_71/kernel/Read/ReadVariableOp"conv2d_71/bias/Read/ReadVariableOp0batch_normalization_82/gamma/Read/ReadVariableOp/batch_normalization_82/beta/Read/ReadVariableOp6batch_normalization_82/moving_mean/Read/ReadVariableOp:batch_normalization_82/moving_variance/Read/ReadVariableOp0batch_normalization_83/gamma/Read/ReadVariableOp/batch_normalization_83/beta/Read/ReadVariableOp6batch_normalization_83/moving_mean/Read/ReadVariableOp:batch_normalization_83/moving_variance/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpConst*7
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
__inference__traced_save_696241
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_66/kernelconv2d_66/biasbatch_normalization_77/gammabatch_normalization_77/beta"batch_normalization_77/moving_mean&batch_normalization_77/moving_varianceconv2d_67/kernelconv2d_67/biasbatch_normalization_78/gammabatch_normalization_78/beta"batch_normalization_78/moving_mean&batch_normalization_78/moving_varianceconv2d_68/kernelconv2d_68/biasbatch_normalization_79/gammabatch_normalization_79/beta"batch_normalization_79/moving_mean&batch_normalization_79/moving_varianceconv2d_69/kernelconv2d_69/biasbatch_normalization_80/gammabatch_normalization_80/beta"batch_normalization_80/moving_mean&batch_normalization_80/moving_varianceconv2d_70/kernelconv2d_70/biasbatch_normalization_81/gammabatch_normalization_81/beta"batch_normalization_81/moving_mean&batch_normalization_81/moving_varianceconv2d_71/kernelconv2d_71/biasbatch_normalization_82/gammabatch_normalization_82/beta"batch_normalization_82/moving_mean&batch_normalization_82/moving_variancebatch_normalization_83/gammabatch_normalization_83/beta"batch_normalization_83/moving_mean&batch_normalization_83/moving_variancedense_11/kerneldense_11/bias*6
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
"__inference__traced_restore_696377��
�

�
E__inference_conv2d_67_layer_call_and_return_conditional_losses_695172

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
�
�
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695497

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
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_693020

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
7__inference_batch_normalization_79_layer_call_fn_695457

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_6934162
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
7__inference_batch_normalization_82_layer_call_fn_695837

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_6930202
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
7__inference_batch_normalization_83_layer_call_fn_695965

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_6931362
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
�

�
E__inference_conv2d_69_layer_call_and_return_conditional_losses_695468

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
�
�
7__inference_batch_normalization_81_layer_call_fn_695753

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_6936172
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
7__inference_batch_normalization_83_layer_call_fn_696029

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_6937912
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
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_695921

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
ҳ
�
"__inference__traced_restore_696377
file_prefix%
!assignvariableop_conv2d_66_kernel%
!assignvariableop_1_conv2d_66_bias3
/assignvariableop_2_batch_normalization_77_gamma2
.assignvariableop_3_batch_normalization_77_beta9
5assignvariableop_4_batch_normalization_77_moving_mean=
9assignvariableop_5_batch_normalization_77_moving_variance'
#assignvariableop_6_conv2d_67_kernel%
!assignvariableop_7_conv2d_67_bias3
/assignvariableop_8_batch_normalization_78_gamma2
.assignvariableop_9_batch_normalization_78_beta:
6assignvariableop_10_batch_normalization_78_moving_mean>
:assignvariableop_11_batch_normalization_78_moving_variance(
$assignvariableop_12_conv2d_68_kernel&
"assignvariableop_13_conv2d_68_bias4
0assignvariableop_14_batch_normalization_79_gamma3
/assignvariableop_15_batch_normalization_79_beta:
6assignvariableop_16_batch_normalization_79_moving_mean>
:assignvariableop_17_batch_normalization_79_moving_variance(
$assignvariableop_18_conv2d_69_kernel&
"assignvariableop_19_conv2d_69_bias4
0assignvariableop_20_batch_normalization_80_gamma3
/assignvariableop_21_batch_normalization_80_beta:
6assignvariableop_22_batch_normalization_80_moving_mean>
:assignvariableop_23_batch_normalization_80_moving_variance(
$assignvariableop_24_conv2d_70_kernel&
"assignvariableop_25_conv2d_70_bias4
0assignvariableop_26_batch_normalization_81_gamma3
/assignvariableop_27_batch_normalization_81_beta:
6assignvariableop_28_batch_normalization_81_moving_mean>
:assignvariableop_29_batch_normalization_81_moving_variance(
$assignvariableop_30_conv2d_71_kernel&
"assignvariableop_31_conv2d_71_bias4
0assignvariableop_32_batch_normalization_82_gamma3
/assignvariableop_33_batch_normalization_82_beta:
6assignvariableop_34_batch_normalization_82_moving_mean>
:assignvariableop_35_batch_normalization_82_moving_variance4
0assignvariableop_36_batch_normalization_83_gamma3
/assignvariableop_37_batch_normalization_83_beta:
6assignvariableop_38_batch_normalization_83_moving_mean>
:assignvariableop_39_batch_normalization_83_moving_variance'
#assignvariableop_40_dense_11_kernel%
!assignvariableop_41_dense_11_bias
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_66_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_66_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_77_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_77_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_77_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_77_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_67_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_67_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_78_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_78_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_78_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_78_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_68_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_68_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_79_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_79_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_79_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_79_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_69_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_69_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_80_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_80_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_80_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_80_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_70_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_70_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_81_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_81_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_81_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_81_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_71_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_71_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_82_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_82_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_82_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_82_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_83_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp/assignvariableop_37_batch_normalization_83_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp6assignvariableop_38_batch_normalization_83_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp:assignvariableop_39_batch_normalization_83_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp#assignvariableop_40_dense_11_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp!assignvariableop_41_dense_11_biasIdentity_41:output:0"/device:CPU:0*
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
�
M
1__inference_max_pooling2d_34_layer_call_fn_692823

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
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_6928172
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
�Z
�
__inference__traced_save_696241
file_prefix/
+savev2_conv2d_66_kernel_read_readvariableop-
)savev2_conv2d_66_bias_read_readvariableop;
7savev2_batch_normalization_77_gamma_read_readvariableop:
6savev2_batch_normalization_77_beta_read_readvariableopA
=savev2_batch_normalization_77_moving_mean_read_readvariableopE
Asavev2_batch_normalization_77_moving_variance_read_readvariableop/
+savev2_conv2d_67_kernel_read_readvariableop-
)savev2_conv2d_67_bias_read_readvariableop;
7savev2_batch_normalization_78_gamma_read_readvariableop:
6savev2_batch_normalization_78_beta_read_readvariableopA
=savev2_batch_normalization_78_moving_mean_read_readvariableopE
Asavev2_batch_normalization_78_moving_variance_read_readvariableop/
+savev2_conv2d_68_kernel_read_readvariableop-
)savev2_conv2d_68_bias_read_readvariableop;
7savev2_batch_normalization_79_gamma_read_readvariableop:
6savev2_batch_normalization_79_beta_read_readvariableopA
=savev2_batch_normalization_79_moving_mean_read_readvariableopE
Asavev2_batch_normalization_79_moving_variance_read_readvariableop/
+savev2_conv2d_69_kernel_read_readvariableop-
)savev2_conv2d_69_bias_read_readvariableop;
7savev2_batch_normalization_80_gamma_read_readvariableop:
6savev2_batch_normalization_80_beta_read_readvariableopA
=savev2_batch_normalization_80_moving_mean_read_readvariableopE
Asavev2_batch_normalization_80_moving_variance_read_readvariableop/
+savev2_conv2d_70_kernel_read_readvariableop-
)savev2_conv2d_70_bias_read_readvariableop;
7savev2_batch_normalization_81_gamma_read_readvariableop:
6savev2_batch_normalization_81_beta_read_readvariableopA
=savev2_batch_normalization_81_moving_mean_read_readvariableopE
Asavev2_batch_normalization_81_moving_variance_read_readvariableop/
+savev2_conv2d_71_kernel_read_readvariableop-
)savev2_conv2d_71_bias_read_readvariableop;
7savev2_batch_normalization_82_gamma_read_readvariableop:
6savev2_batch_normalization_82_beta_read_readvariableopA
=savev2_batch_normalization_82_moving_mean_read_readvariableopE
Asavev2_batch_normalization_82_moving_variance_read_readvariableop;
7savev2_batch_normalization_83_gamma_read_readvariableop:
6savev2_batch_normalization_83_beta_read_readvariableopA
=savev2_batch_normalization_83_moving_mean_read_readvariableopE
Asavev2_batch_normalization_83_moving_variance_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop
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
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_66_kernel_read_readvariableop)savev2_conv2d_66_bias_read_readvariableop7savev2_batch_normalization_77_gamma_read_readvariableop6savev2_batch_normalization_77_beta_read_readvariableop=savev2_batch_normalization_77_moving_mean_read_readvariableopAsavev2_batch_normalization_77_moving_variance_read_readvariableop+savev2_conv2d_67_kernel_read_readvariableop)savev2_conv2d_67_bias_read_readvariableop7savev2_batch_normalization_78_gamma_read_readvariableop6savev2_batch_normalization_78_beta_read_readvariableop=savev2_batch_normalization_78_moving_mean_read_readvariableopAsavev2_batch_normalization_78_moving_variance_read_readvariableop+savev2_conv2d_68_kernel_read_readvariableop)savev2_conv2d_68_bias_read_readvariableop7savev2_batch_normalization_79_gamma_read_readvariableop6savev2_batch_normalization_79_beta_read_readvariableop=savev2_batch_normalization_79_moving_mean_read_readvariableopAsavev2_batch_normalization_79_moving_variance_read_readvariableop+savev2_conv2d_69_kernel_read_readvariableop)savev2_conv2d_69_bias_read_readvariableop7savev2_batch_normalization_80_gamma_read_readvariableop6savev2_batch_normalization_80_beta_read_readvariableop=savev2_batch_normalization_80_moving_mean_read_readvariableopAsavev2_batch_normalization_80_moving_variance_read_readvariableop+savev2_conv2d_70_kernel_read_readvariableop)savev2_conv2d_70_bias_read_readvariableop7savev2_batch_normalization_81_gamma_read_readvariableop6savev2_batch_normalization_81_beta_read_readvariableop=savev2_batch_normalization_81_moving_mean_read_readvariableopAsavev2_batch_normalization_81_moving_variance_read_readvariableop+savev2_conv2d_71_kernel_read_readvariableop)savev2_conv2d_71_bias_read_readvariableop7savev2_batch_normalization_82_gamma_read_readvariableop6savev2_batch_normalization_82_beta_read_readvariableop=savev2_batch_normalization_82_moving_mean_read_readvariableopAsavev2_batch_normalization_82_moving_variance_read_readvariableop7savev2_batch_normalization_83_gamma_read_readvariableop6savev2_batch_normalization_83_beta_read_readvariableop=savev2_batch_normalization_83_moving_mean_read_readvariableopAsavev2_batch_normalization_83_moving_variance_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�

�
E__inference_conv2d_66_layer_call_and_return_conditional_losses_695024

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
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695219

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
7__inference_batch_normalization_78_layer_call_fn_695232

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_6925492
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
�
�
7__inference_batch_normalization_80_layer_call_fn_695592

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_6934982
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
��
�&
I__inference_sequential_11_layer_call_and_return_conditional_losses_694672

inputs,
(conv2d_66_conv2d_readvariableop_resource-
)conv2d_66_biasadd_readvariableop_resource2
.batch_normalization_77_readvariableop_resource4
0batch_normalization_77_readvariableop_1_resourceC
?batch_normalization_77_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_67_conv2d_readvariableop_resource-
)conv2d_67_biasadd_readvariableop_resource2
.batch_normalization_78_readvariableop_resource4
0batch_normalization_78_readvariableop_1_resourceC
?batch_normalization_78_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_78_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_68_conv2d_readvariableop_resource-
)conv2d_68_biasadd_readvariableop_resource2
.batch_normalization_79_readvariableop_resource4
0batch_normalization_79_readvariableop_1_resourceC
?batch_normalization_79_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_79_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_69_conv2d_readvariableop_resource-
)conv2d_69_biasadd_readvariableop_resource2
.batch_normalization_80_readvariableop_resource4
0batch_normalization_80_readvariableop_1_resourceC
?batch_normalization_80_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_80_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_70_conv2d_readvariableop_resource-
)conv2d_70_biasadd_readvariableop_resource2
.batch_normalization_81_readvariableop_resource4
0batch_normalization_81_readvariableop_1_resourceC
?batch_normalization_81_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_81_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_71_conv2d_readvariableop_resource-
)conv2d_71_biasadd_readvariableop_resource2
.batch_normalization_82_readvariableop_resource4
0batch_normalization_82_readvariableop_1_resourceC
?batch_normalization_82_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_82_fusedbatchnormv3_readvariableop_1_resource2
.batch_normalization_83_readvariableop_resource4
0batch_normalization_83_readvariableop_1_resourceC
?batch_normalization_83_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_83_fusedbatchnormv3_readvariableop_1_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity��%batch_normalization_77/AssignNewValue�'batch_normalization_77/AssignNewValue_1�6batch_normalization_77/FusedBatchNormV3/ReadVariableOp�8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_77/ReadVariableOp�'batch_normalization_77/ReadVariableOp_1�%batch_normalization_78/AssignNewValue�'batch_normalization_78/AssignNewValue_1�6batch_normalization_78/FusedBatchNormV3/ReadVariableOp�8batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_78/ReadVariableOp�'batch_normalization_78/ReadVariableOp_1�%batch_normalization_79/AssignNewValue�'batch_normalization_79/AssignNewValue_1�6batch_normalization_79/FusedBatchNormV3/ReadVariableOp�8batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_79/ReadVariableOp�'batch_normalization_79/ReadVariableOp_1�%batch_normalization_80/AssignNewValue�'batch_normalization_80/AssignNewValue_1�6batch_normalization_80/FusedBatchNormV3/ReadVariableOp�8batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_80/ReadVariableOp�'batch_normalization_80/ReadVariableOp_1�%batch_normalization_81/AssignNewValue�'batch_normalization_81/AssignNewValue_1�6batch_normalization_81/FusedBatchNormV3/ReadVariableOp�8batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_81/ReadVariableOp�'batch_normalization_81/ReadVariableOp_1�%batch_normalization_82/AssignNewValue�'batch_normalization_82/AssignNewValue_1�6batch_normalization_82/FusedBatchNormV3/ReadVariableOp�8batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_82/ReadVariableOp�'batch_normalization_82/ReadVariableOp_1�%batch_normalization_83/AssignNewValue�'batch_normalization_83/AssignNewValue_1�6batch_normalization_83/FusedBatchNormV3/ReadVariableOp�8batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_83/ReadVariableOp�'batch_normalization_83/ReadVariableOp_1� conv2d_66/BiasAdd/ReadVariableOp�conv2d_66/Conv2D/ReadVariableOp� conv2d_67/BiasAdd/ReadVariableOp�conv2d_67/Conv2D/ReadVariableOp� conv2d_68/BiasAdd/ReadVariableOp�conv2d_68/Conv2D/ReadVariableOp� conv2d_69/BiasAdd/ReadVariableOp�conv2d_69/Conv2D/ReadVariableOp� conv2d_70/BiasAdd/ReadVariableOp�conv2d_70/Conv2D/ReadVariableOp� conv2d_71/BiasAdd/ReadVariableOp�conv2d_71/Conv2D/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_66/Conv2D/ReadVariableOp�
conv2d_66/Conv2DConv2Dinputs'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_66/Conv2D�
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_66/BiasAdd/ReadVariableOp�
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_66/BiasAdd~
conv2d_66/ReluReluconv2d_66/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_66/Relu�
%batch_normalization_77/ReadVariableOpReadVariableOp.batch_normalization_77_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_77/ReadVariableOp�
'batch_normalization_77/ReadVariableOp_1ReadVariableOp0batch_normalization_77_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_77/ReadVariableOp_1�
6batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_77/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_77/FusedBatchNormV3FusedBatchNormV3conv2d_66/Relu:activations:0-batch_normalization_77/ReadVariableOp:value:0/batch_normalization_77/ReadVariableOp_1:value:0>batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_77/FusedBatchNormV3�
%batch_normalization_77/AssignNewValueAssignVariableOp?batch_normalization_77_fusedbatchnormv3_readvariableop_resource4batch_normalization_77/FusedBatchNormV3:batch_mean:07^batch_normalization_77/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_77/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_77/AssignNewValue�
'batch_normalization_77/AssignNewValue_1AssignVariableOpAbatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_77/FusedBatchNormV3:batch_variance:09^batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_77/AssignNewValue_1�
conv2d_67/Conv2D/ReadVariableOpReadVariableOp(conv2d_67_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_67/Conv2D/ReadVariableOp�
conv2d_67/Conv2DConv2D+batch_normalization_77/FusedBatchNormV3:y:0'conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_67/Conv2D�
 conv2d_67/BiasAdd/ReadVariableOpReadVariableOp)conv2d_67_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_67/BiasAdd/ReadVariableOp�
conv2d_67/BiasAddBiasAddconv2d_67/Conv2D:output:0(conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_67/BiasAdd~
conv2d_67/ReluReluconv2d_67/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_67/Relu�
%batch_normalization_78/ReadVariableOpReadVariableOp.batch_normalization_78_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_78/ReadVariableOp�
'batch_normalization_78/ReadVariableOp_1ReadVariableOp0batch_normalization_78_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_78/ReadVariableOp_1�
6batch_normalization_78/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_78_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_78/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_78_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_78/FusedBatchNormV3FusedBatchNormV3conv2d_67/Relu:activations:0-batch_normalization_78/ReadVariableOp:value:0/batch_normalization_78/ReadVariableOp_1:value:0>batch_normalization_78/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_78/FusedBatchNormV3�
%batch_normalization_78/AssignNewValueAssignVariableOp?batch_normalization_78_fusedbatchnormv3_readvariableop_resource4batch_normalization_78/FusedBatchNormV3:batch_mean:07^batch_normalization_78/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_78/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_78/AssignNewValue�
'batch_normalization_78/AssignNewValue_1AssignVariableOpAbatch_normalization_78_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_78/FusedBatchNormV3:batch_variance:09^batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_78/AssignNewValue_1�
max_pooling2d_33/MaxPoolMaxPool+batch_normalization_78/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_33/MaxPool�
conv2d_68/Conv2D/ReadVariableOpReadVariableOp(conv2d_68_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_68/Conv2D/ReadVariableOp�
conv2d_68/Conv2DConv2D!max_pooling2d_33/MaxPool:output:0'conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_68/Conv2D�
 conv2d_68/BiasAdd/ReadVariableOpReadVariableOp)conv2d_68_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_68/BiasAdd/ReadVariableOp�
conv2d_68/BiasAddBiasAddconv2d_68/Conv2D:output:0(conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_68/BiasAdd~
conv2d_68/ReluReluconv2d_68/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_68/Relu�
%batch_normalization_79/ReadVariableOpReadVariableOp.batch_normalization_79_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_79/ReadVariableOp�
'batch_normalization_79/ReadVariableOp_1ReadVariableOp0batch_normalization_79_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_79/ReadVariableOp_1�
6batch_normalization_79/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_79_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_79/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_79_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_79/FusedBatchNormV3FusedBatchNormV3conv2d_68/Relu:activations:0-batch_normalization_79/ReadVariableOp:value:0/batch_normalization_79/ReadVariableOp_1:value:0>batch_normalization_79/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_79/FusedBatchNormV3�
%batch_normalization_79/AssignNewValueAssignVariableOp?batch_normalization_79_fusedbatchnormv3_readvariableop_resource4batch_normalization_79/FusedBatchNormV3:batch_mean:07^batch_normalization_79/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_79/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_79/AssignNewValue�
'batch_normalization_79/AssignNewValue_1AssignVariableOpAbatch_normalization_79_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_79/FusedBatchNormV3:batch_variance:09^batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_79/AssignNewValue_1�
conv2d_69/Conv2D/ReadVariableOpReadVariableOp(conv2d_69_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_69/Conv2D/ReadVariableOp�
conv2d_69/Conv2DConv2D+batch_normalization_79/FusedBatchNormV3:y:0'conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_69/Conv2D�
 conv2d_69/BiasAdd/ReadVariableOpReadVariableOp)conv2d_69_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_69/BiasAdd/ReadVariableOp�
conv2d_69/BiasAddBiasAddconv2d_69/Conv2D:output:0(conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_69/BiasAdd~
conv2d_69/ReluReluconv2d_69/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_69/Relu�
%batch_normalization_80/ReadVariableOpReadVariableOp.batch_normalization_80_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_80/ReadVariableOp�
'batch_normalization_80/ReadVariableOp_1ReadVariableOp0batch_normalization_80_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_80/ReadVariableOp_1�
6batch_normalization_80/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_80_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_80/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_80_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_80/FusedBatchNormV3FusedBatchNormV3conv2d_69/Relu:activations:0-batch_normalization_80/ReadVariableOp:value:0/batch_normalization_80/ReadVariableOp_1:value:0>batch_normalization_80/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_80/FusedBatchNormV3�
%batch_normalization_80/AssignNewValueAssignVariableOp?batch_normalization_80_fusedbatchnormv3_readvariableop_resource4batch_normalization_80/FusedBatchNormV3:batch_mean:07^batch_normalization_80/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_80/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_80/AssignNewValue�
'batch_normalization_80/AssignNewValue_1AssignVariableOpAbatch_normalization_80_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_80/FusedBatchNormV3:batch_variance:09^batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_80/AssignNewValue_1�
max_pooling2d_34/MaxPoolMaxPool+batch_normalization_80/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_34/MaxPool�
conv2d_70/Conv2D/ReadVariableOpReadVariableOp(conv2d_70_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_70/Conv2D/ReadVariableOp�
conv2d_70/Conv2DConv2D!max_pooling2d_34/MaxPool:output:0'conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_70/Conv2D�
 conv2d_70/BiasAdd/ReadVariableOpReadVariableOp)conv2d_70_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_70/BiasAdd/ReadVariableOp�
conv2d_70/BiasAddBiasAddconv2d_70/Conv2D:output:0(conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_70/BiasAdd
conv2d_70/ReluReluconv2d_70/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_70/Relu�
%batch_normalization_81/ReadVariableOpReadVariableOp.batch_normalization_81_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_81/ReadVariableOp�
'batch_normalization_81/ReadVariableOp_1ReadVariableOp0batch_normalization_81_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_81/ReadVariableOp_1�
6batch_normalization_81/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_81_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_81/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_81_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_81/FusedBatchNormV3FusedBatchNormV3conv2d_70/Relu:activations:0-batch_normalization_81/ReadVariableOp:value:0/batch_normalization_81/ReadVariableOp_1:value:0>batch_normalization_81/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_81/FusedBatchNormV3�
%batch_normalization_81/AssignNewValueAssignVariableOp?batch_normalization_81_fusedbatchnormv3_readvariableop_resource4batch_normalization_81/FusedBatchNormV3:batch_mean:07^batch_normalization_81/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_81/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_81/AssignNewValue�
'batch_normalization_81/AssignNewValue_1AssignVariableOpAbatch_normalization_81_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_81/FusedBatchNormV3:batch_variance:09^batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_81/AssignNewValue_1�
conv2d_71/Conv2D/ReadVariableOpReadVariableOp(conv2d_71_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_71/Conv2D/ReadVariableOp�
conv2d_71/Conv2DConv2D+batch_normalization_81/FusedBatchNormV3:y:0'conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_71/Conv2D�
 conv2d_71/BiasAdd/ReadVariableOpReadVariableOp)conv2d_71_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_71/BiasAdd/ReadVariableOp�
conv2d_71/BiasAddBiasAddconv2d_71/Conv2D:output:0(conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_71/BiasAdd
conv2d_71/ReluReluconv2d_71/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_71/Relu�
%batch_normalization_82/ReadVariableOpReadVariableOp.batch_normalization_82_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_82/ReadVariableOp�
'batch_normalization_82/ReadVariableOp_1ReadVariableOp0batch_normalization_82_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_82/ReadVariableOp_1�
6batch_normalization_82/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_82_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_82/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_82_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_82/FusedBatchNormV3FusedBatchNormV3conv2d_71/Relu:activations:0-batch_normalization_82/ReadVariableOp:value:0/batch_normalization_82/ReadVariableOp_1:value:0>batch_normalization_82/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_82/FusedBatchNormV3�
%batch_normalization_82/AssignNewValueAssignVariableOp?batch_normalization_82_fusedbatchnormv3_readvariableop_resource4batch_normalization_82/FusedBatchNormV3:batch_mean:07^batch_normalization_82/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_82/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_82/AssignNewValue�
'batch_normalization_82/AssignNewValue_1AssignVariableOpAbatch_normalization_82_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_82/FusedBatchNormV3:batch_variance:09^batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_82/AssignNewValue_1�
max_pooling2d_35/MaxPoolMaxPool+batch_normalization_82/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_35/MaxPool�
%batch_normalization_83/ReadVariableOpReadVariableOp.batch_normalization_83_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_83/ReadVariableOp�
'batch_normalization_83/ReadVariableOp_1ReadVariableOp0batch_normalization_83_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_83/ReadVariableOp_1�
6batch_normalization_83/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_83_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_83/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_83_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_83/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_35/MaxPool:output:0-batch_normalization_83/ReadVariableOp:value:0/batch_normalization_83/ReadVariableOp_1:value:0>batch_normalization_83/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_83/FusedBatchNormV3�
%batch_normalization_83/AssignNewValueAssignVariableOp?batch_normalization_83_fusedbatchnormv3_readvariableop_resource4batch_normalization_83/FusedBatchNormV3:batch_mean:07^batch_normalization_83/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_83/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_83/AssignNewValue�
'batch_normalization_83/AssignNewValue_1AssignVariableOpAbatch_normalization_83_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_83/FusedBatchNormV3:batch_variance:09^batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_83/AssignNewValue_1u
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_11/Const�
flatten_11/ReshapeReshape+batch_normalization_83/FusedBatchNormV3:y:0flatten_11/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_11/Reshape�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMulflatten_11/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_11/BiasAdd}
dense_11/SigmoidSigmoiddense_11/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_11/Sigmoid�
lambda_11/l2_normalize/SquareSquaredense_11/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
lambda_11/l2_normalize/Square�
,lambda_11/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,lambda_11/l2_normalize/Sum/reduction_indices�
lambda_11/l2_normalize/SumSum!lambda_11/l2_normalize/Square:y:05lambda_11/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
lambda_11/l2_normalize/Sum�
 lambda_11/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2"
 lambda_11/l2_normalize/Maximum/y�
lambda_11/l2_normalize/MaximumMaximum#lambda_11/l2_normalize/Sum:output:0)lambda_11/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2 
lambda_11/l2_normalize/Maximum�
lambda_11/l2_normalize/RsqrtRsqrt"lambda_11/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
lambda_11/l2_normalize/Rsqrt�
lambda_11/l2_normalizeMuldense_11/Sigmoid:y:0 lambda_11/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
lambda_11/l2_normalize�
IdentityIdentitylambda_11/l2_normalize:z:0&^batch_normalization_77/AssignNewValue(^batch_normalization_77/AssignNewValue_17^batch_normalization_77/FusedBatchNormV3/ReadVariableOp9^batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_77/ReadVariableOp(^batch_normalization_77/ReadVariableOp_1&^batch_normalization_78/AssignNewValue(^batch_normalization_78/AssignNewValue_17^batch_normalization_78/FusedBatchNormV3/ReadVariableOp9^batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_78/ReadVariableOp(^batch_normalization_78/ReadVariableOp_1&^batch_normalization_79/AssignNewValue(^batch_normalization_79/AssignNewValue_17^batch_normalization_79/FusedBatchNormV3/ReadVariableOp9^batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_79/ReadVariableOp(^batch_normalization_79/ReadVariableOp_1&^batch_normalization_80/AssignNewValue(^batch_normalization_80/AssignNewValue_17^batch_normalization_80/FusedBatchNormV3/ReadVariableOp9^batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_80/ReadVariableOp(^batch_normalization_80/ReadVariableOp_1&^batch_normalization_81/AssignNewValue(^batch_normalization_81/AssignNewValue_17^batch_normalization_81/FusedBatchNormV3/ReadVariableOp9^batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_81/ReadVariableOp(^batch_normalization_81/ReadVariableOp_1&^batch_normalization_82/AssignNewValue(^batch_normalization_82/AssignNewValue_17^batch_normalization_82/FusedBatchNormV3/ReadVariableOp9^batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_82/ReadVariableOp(^batch_normalization_82/ReadVariableOp_1&^batch_normalization_83/AssignNewValue(^batch_normalization_83/AssignNewValue_17^batch_normalization_83/FusedBatchNormV3/ReadVariableOp9^batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_83/ReadVariableOp(^batch_normalization_83/ReadVariableOp_1!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp!^conv2d_67/BiasAdd/ReadVariableOp ^conv2d_67/Conv2D/ReadVariableOp!^conv2d_68/BiasAdd/ReadVariableOp ^conv2d_68/Conv2D/ReadVariableOp!^conv2d_69/BiasAdd/ReadVariableOp ^conv2d_69/Conv2D/ReadVariableOp!^conv2d_70/BiasAdd/ReadVariableOp ^conv2d_70/Conv2D/ReadVariableOp!^conv2d_71/BiasAdd/ReadVariableOp ^conv2d_71/Conv2D/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_77/AssignNewValue%batch_normalization_77/AssignNewValue2R
'batch_normalization_77/AssignNewValue_1'batch_normalization_77/AssignNewValue_12p
6batch_normalization_77/FusedBatchNormV3/ReadVariableOp6batch_normalization_77/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_18batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_77/ReadVariableOp%batch_normalization_77/ReadVariableOp2R
'batch_normalization_77/ReadVariableOp_1'batch_normalization_77/ReadVariableOp_12N
%batch_normalization_78/AssignNewValue%batch_normalization_78/AssignNewValue2R
'batch_normalization_78/AssignNewValue_1'batch_normalization_78/AssignNewValue_12p
6batch_normalization_78/FusedBatchNormV3/ReadVariableOp6batch_normalization_78/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_78/FusedBatchNormV3/ReadVariableOp_18batch_normalization_78/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_78/ReadVariableOp%batch_normalization_78/ReadVariableOp2R
'batch_normalization_78/ReadVariableOp_1'batch_normalization_78/ReadVariableOp_12N
%batch_normalization_79/AssignNewValue%batch_normalization_79/AssignNewValue2R
'batch_normalization_79/AssignNewValue_1'batch_normalization_79/AssignNewValue_12p
6batch_normalization_79/FusedBatchNormV3/ReadVariableOp6batch_normalization_79/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_79/FusedBatchNormV3/ReadVariableOp_18batch_normalization_79/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_79/ReadVariableOp%batch_normalization_79/ReadVariableOp2R
'batch_normalization_79/ReadVariableOp_1'batch_normalization_79/ReadVariableOp_12N
%batch_normalization_80/AssignNewValue%batch_normalization_80/AssignNewValue2R
'batch_normalization_80/AssignNewValue_1'batch_normalization_80/AssignNewValue_12p
6batch_normalization_80/FusedBatchNormV3/ReadVariableOp6batch_normalization_80/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_80/FusedBatchNormV3/ReadVariableOp_18batch_normalization_80/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_80/ReadVariableOp%batch_normalization_80/ReadVariableOp2R
'batch_normalization_80/ReadVariableOp_1'batch_normalization_80/ReadVariableOp_12N
%batch_normalization_81/AssignNewValue%batch_normalization_81/AssignNewValue2R
'batch_normalization_81/AssignNewValue_1'batch_normalization_81/AssignNewValue_12p
6batch_normalization_81/FusedBatchNormV3/ReadVariableOp6batch_normalization_81/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_81/FusedBatchNormV3/ReadVariableOp_18batch_normalization_81/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_81/ReadVariableOp%batch_normalization_81/ReadVariableOp2R
'batch_normalization_81/ReadVariableOp_1'batch_normalization_81/ReadVariableOp_12N
%batch_normalization_82/AssignNewValue%batch_normalization_82/AssignNewValue2R
'batch_normalization_82/AssignNewValue_1'batch_normalization_82/AssignNewValue_12p
6batch_normalization_82/FusedBatchNormV3/ReadVariableOp6batch_normalization_82/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_82/FusedBatchNormV3/ReadVariableOp_18batch_normalization_82/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_82/ReadVariableOp%batch_normalization_82/ReadVariableOp2R
'batch_normalization_82/ReadVariableOp_1'batch_normalization_82/ReadVariableOp_12N
%batch_normalization_83/AssignNewValue%batch_normalization_83/AssignNewValue2R
'batch_normalization_83/AssignNewValue_1'batch_normalization_83/AssignNewValue_12p
6batch_normalization_83/FusedBatchNormV3/ReadVariableOp6batch_normalization_83/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_83/FusedBatchNormV3/ReadVariableOp_18batch_normalization_83/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_83/ReadVariableOp%batch_normalization_83/ReadVariableOp2R
'batch_normalization_83/ReadVariableOp_1'batch_normalization_83/ReadVariableOp_12D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2D
 conv2d_67/BiasAdd/ReadVariableOp conv2d_67/BiasAdd/ReadVariableOp2B
conv2d_67/Conv2D/ReadVariableOpconv2d_67/Conv2D/ReadVariableOp2D
 conv2d_68/BiasAdd/ReadVariableOp conv2d_68/BiasAdd/ReadVariableOp2B
conv2d_68/Conv2D/ReadVariableOpconv2d_68/Conv2D/ReadVariableOp2D
 conv2d_69/BiasAdd/ReadVariableOp conv2d_69/BiasAdd/ReadVariableOp2B
conv2d_69/Conv2D/ReadVariableOpconv2d_69/Conv2D/ReadVariableOp2D
 conv2d_70/BiasAdd/ReadVariableOp conv2d_70/BiasAdd/ReadVariableOp2B
conv2d_70/Conv2D/ReadVariableOpconv2d_70/Conv2D/ReadVariableOp2D
 conv2d_71/BiasAdd/ReadVariableOp conv2d_71/BiasAdd/ReadVariableOp2B
conv2d_71/Conv2D/ReadVariableOpconv2d_71/Conv2D/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695645

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
7__inference_batch_normalization_78_layer_call_fn_695309

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_6933152
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
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695265

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
�j
�
I__inference_sequential_11_layer_call_and_return_conditional_losses_693904
conv2d_66_input
conv2d_66_693173
conv2d_66_693175!
batch_normalization_77_693242!
batch_normalization_77_693244!
batch_normalization_77_693246!
batch_normalization_77_693248
conv2d_67_693273
conv2d_67_693275!
batch_normalization_78_693342!
batch_normalization_78_693344!
batch_normalization_78_693346!
batch_normalization_78_693348
conv2d_68_693374
conv2d_68_693376!
batch_normalization_79_693443!
batch_normalization_79_693445!
batch_normalization_79_693447!
batch_normalization_79_693449
conv2d_69_693474
conv2d_69_693476!
batch_normalization_80_693543!
batch_normalization_80_693545!
batch_normalization_80_693547!
batch_normalization_80_693549
conv2d_70_693575
conv2d_70_693577!
batch_normalization_81_693644!
batch_normalization_81_693646!
batch_normalization_81_693648!
batch_normalization_81_693650
conv2d_71_693675
conv2d_71_693677!
batch_normalization_82_693744!
batch_normalization_82_693746!
batch_normalization_82_693748!
batch_normalization_82_693750!
batch_normalization_83_693818!
batch_normalization_83_693820!
batch_normalization_83_693822!
batch_normalization_83_693824
dense_11_693863
dense_11_693865
identity��.batch_normalization_77/StatefulPartitionedCall�.batch_normalization_78/StatefulPartitionedCall�.batch_normalization_79/StatefulPartitionedCall�.batch_normalization_80/StatefulPartitionedCall�.batch_normalization_81/StatefulPartitionedCall�.batch_normalization_82/StatefulPartitionedCall�.batch_normalization_83/StatefulPartitionedCall�!conv2d_66/StatefulPartitionedCall�!conv2d_67/StatefulPartitionedCall�!conv2d_68/StatefulPartitionedCall�!conv2d_69/StatefulPartitionedCall�!conv2d_70/StatefulPartitionedCall�!conv2d_71/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallconv2d_66_inputconv2d_66_693173conv2d_66_693175*
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
E__inference_conv2d_66_layer_call_and_return_conditional_losses_6931622#
!conv2d_66/StatefulPartitionedCall�
.batch_normalization_77/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_77_693242batch_normalization_77_693244batch_normalization_77_693246batch_normalization_77_693248*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_69319720
.batch_normalization_77/StatefulPartitionedCall�
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_77/StatefulPartitionedCall:output:0conv2d_67_693273conv2d_67_693275*
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
E__inference_conv2d_67_layer_call_and_return_conditional_losses_6932622#
!conv2d_67/StatefulPartitionedCall�
.batch_normalization_78/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_78_693342batch_normalization_78_693344batch_normalization_78_693346batch_normalization_78_693348*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_69329720
.batch_normalization_78/StatefulPartitionedCall�
 max_pooling2d_33/PartitionedCallPartitionedCall7batch_normalization_78/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_6925972"
 max_pooling2d_33/PartitionedCall�
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0conv2d_68_693374conv2d_68_693376*
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
E__inference_conv2d_68_layer_call_and_return_conditional_losses_6933632#
!conv2d_68/StatefulPartitionedCall�
.batch_normalization_79/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_79_693443batch_normalization_79_693445batch_normalization_79_693447batch_normalization_79_693449*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_69339820
.batch_normalization_79/StatefulPartitionedCall�
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_79/StatefulPartitionedCall:output:0conv2d_69_693474conv2d_69_693476*
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
E__inference_conv2d_69_layer_call_and_return_conditional_losses_6934632#
!conv2d_69/StatefulPartitionedCall�
.batch_normalization_80/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0batch_normalization_80_693543batch_normalization_80_693545batch_normalization_80_693547batch_normalization_80_693549*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_69349820
.batch_normalization_80/StatefulPartitionedCall�
 max_pooling2d_34/PartitionedCallPartitionedCall7batch_normalization_80/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_6928172"
 max_pooling2d_34/PartitionedCall�
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0conv2d_70_693575conv2d_70_693577*
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
E__inference_conv2d_70_layer_call_and_return_conditional_losses_6935642#
!conv2d_70/StatefulPartitionedCall�
.batch_normalization_81/StatefulPartitionedCallStatefulPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0batch_normalization_81_693644batch_normalization_81_693646batch_normalization_81_693648batch_normalization_81_693650*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_69359920
.batch_normalization_81/StatefulPartitionedCall�
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_81/StatefulPartitionedCall:output:0conv2d_71_693675conv2d_71_693677*
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
E__inference_conv2d_71_layer_call_and_return_conditional_losses_6936642#
!conv2d_71/StatefulPartitionedCall�
.batch_normalization_82/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0batch_normalization_82_693744batch_normalization_82_693746batch_normalization_82_693748batch_normalization_82_693750*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_69369920
.batch_normalization_82/StatefulPartitionedCall�
 max_pooling2d_35/PartitionedCallPartitionedCall7batch_normalization_82/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_6930372"
 max_pooling2d_35/PartitionedCall�
.batch_normalization_83/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_83_693818batch_normalization_83_693820batch_normalization_83_693822batch_normalization_83_693824*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_69377320
.batch_normalization_83/StatefulPartitionedCall�
flatten_11/PartitionedCallPartitionedCall7batch_normalization_83/StatefulPartitionedCall:output:0*
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
F__inference_flatten_11_layer_call_and_return_conditional_losses_6938332
flatten_11/PartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_11_693863dense_11_693865*
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
D__inference_dense_11_layer_call_and_return_conditional_losses_6938522"
 dense_11/StatefulPartitionedCall�
lambda_11/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
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
E__inference_lambda_11_layer_call_and_return_conditional_losses_6938792
lambda_11/PartitionedCall�
IdentityIdentity"lambda_11/PartitionedCall:output:0/^batch_normalization_77/StatefulPartitionedCall/^batch_normalization_78/StatefulPartitionedCall/^batch_normalization_79/StatefulPartitionedCall/^batch_normalization_80/StatefulPartitionedCall/^batch_normalization_81/StatefulPartitionedCall/^batch_normalization_82/StatefulPartitionedCall/^batch_normalization_83/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_77/StatefulPartitionedCall.batch_normalization_77/StatefulPartitionedCall2`
.batch_normalization_78/StatefulPartitionedCall.batch_normalization_78/StatefulPartitionedCall2`
.batch_normalization_79/StatefulPartitionedCall.batch_normalization_79/StatefulPartitionedCall2`
.batch_normalization_80/StatefulPartitionedCall.batch_normalization_80/StatefulPartitionedCall2`
.batch_normalization_81/StatefulPartitionedCall.batch_normalization_81/StatefulPartitionedCall2`
.batch_normalization_82/StatefulPartitionedCall.batch_normalization_82/StatefulPartitionedCall2`
.batch_normalization_83/StatefulPartitionedCall.batch_normalization_83/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_66_input
�
�
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_693197

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
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_693136

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
�
�
7__inference_batch_normalization_80_layer_call_fn_695528

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_6927692
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
�
h
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_692817

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
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_696003

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
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_693315

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
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_692445

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
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695071

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
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_693297

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
7__inference_batch_normalization_77_layer_call_fn_695084

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_6931972
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
7__inference_batch_normalization_79_layer_call_fn_695393

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_6926962
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
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_693599

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
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_692769

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
E__inference_conv2d_70_layer_call_and_return_conditional_losses_695616

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
�
�
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695515

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
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695561

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
�

a
E__inference_lambda_11_layer_call_and_return_conditional_losses_696071

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
7__inference_batch_normalization_81_layer_call_fn_695740

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_6935992
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
�	
�
D__inference_dense_11_layer_call_and_return_conditional_losses_693852

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
�

*__inference_conv2d_68_layer_call_fn_695329

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
E__inference_conv2d_68_layer_call_and_return_conditional_losses_6933632
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
.__inference_sequential_11_layer_call_fn_694924

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
I__inference_sequential_11_layer_call_and_return_conditional_losses_6941212
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
�
�
$__inference_signature_wrapper_694495
conv2d_66_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_6923832
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
_user_specified_nameconv2d_66_input
�
�
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_693215

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
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_693699

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
�
�
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_693416

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
7__inference_batch_normalization_78_layer_call_fn_695296

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_6932972
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
D__inference_dense_11_layer_call_and_return_conditional_losses_696051

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
�
�
7__inference_batch_normalization_77_layer_call_fn_695097

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_6932152
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
�

*__inference_conv2d_71_layer_call_fn_695773

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
E__inference_conv2d_71_layer_call_and_return_conditional_losses_6936642
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
�
�
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695431

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
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_692549

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
�
h
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_693037

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
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695811

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
�j
�
I__inference_sequential_11_layer_call_and_return_conditional_losses_694317

inputs
conv2d_66_694213
conv2d_66_694215!
batch_normalization_77_694218!
batch_normalization_77_694220!
batch_normalization_77_694222!
batch_normalization_77_694224
conv2d_67_694227
conv2d_67_694229!
batch_normalization_78_694232!
batch_normalization_78_694234!
batch_normalization_78_694236!
batch_normalization_78_694238
conv2d_68_694242
conv2d_68_694244!
batch_normalization_79_694247!
batch_normalization_79_694249!
batch_normalization_79_694251!
batch_normalization_79_694253
conv2d_69_694256
conv2d_69_694258!
batch_normalization_80_694261!
batch_normalization_80_694263!
batch_normalization_80_694265!
batch_normalization_80_694267
conv2d_70_694271
conv2d_70_694273!
batch_normalization_81_694276!
batch_normalization_81_694278!
batch_normalization_81_694280!
batch_normalization_81_694282
conv2d_71_694285
conv2d_71_694287!
batch_normalization_82_694290!
batch_normalization_82_694292!
batch_normalization_82_694294!
batch_normalization_82_694296!
batch_normalization_83_694300!
batch_normalization_83_694302!
batch_normalization_83_694304!
batch_normalization_83_694306
dense_11_694310
dense_11_694312
identity��.batch_normalization_77/StatefulPartitionedCall�.batch_normalization_78/StatefulPartitionedCall�.batch_normalization_79/StatefulPartitionedCall�.batch_normalization_80/StatefulPartitionedCall�.batch_normalization_81/StatefulPartitionedCall�.batch_normalization_82/StatefulPartitionedCall�.batch_normalization_83/StatefulPartitionedCall�!conv2d_66/StatefulPartitionedCall�!conv2d_67/StatefulPartitionedCall�!conv2d_68/StatefulPartitionedCall�!conv2d_69/StatefulPartitionedCall�!conv2d_70/StatefulPartitionedCall�!conv2d_71/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_66_694213conv2d_66_694215*
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
E__inference_conv2d_66_layer_call_and_return_conditional_losses_6931622#
!conv2d_66/StatefulPartitionedCall�
.batch_normalization_77/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_77_694218batch_normalization_77_694220batch_normalization_77_694222batch_normalization_77_694224*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_69321520
.batch_normalization_77/StatefulPartitionedCall�
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_77/StatefulPartitionedCall:output:0conv2d_67_694227conv2d_67_694229*
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
E__inference_conv2d_67_layer_call_and_return_conditional_losses_6932622#
!conv2d_67/StatefulPartitionedCall�
.batch_normalization_78/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_78_694232batch_normalization_78_694234batch_normalization_78_694236batch_normalization_78_694238*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_69331520
.batch_normalization_78/StatefulPartitionedCall�
 max_pooling2d_33/PartitionedCallPartitionedCall7batch_normalization_78/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_6925972"
 max_pooling2d_33/PartitionedCall�
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0conv2d_68_694242conv2d_68_694244*
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
E__inference_conv2d_68_layer_call_and_return_conditional_losses_6933632#
!conv2d_68/StatefulPartitionedCall�
.batch_normalization_79/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_79_694247batch_normalization_79_694249batch_normalization_79_694251batch_normalization_79_694253*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_69341620
.batch_normalization_79/StatefulPartitionedCall�
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_79/StatefulPartitionedCall:output:0conv2d_69_694256conv2d_69_694258*
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
E__inference_conv2d_69_layer_call_and_return_conditional_losses_6934632#
!conv2d_69/StatefulPartitionedCall�
.batch_normalization_80/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0batch_normalization_80_694261batch_normalization_80_694263batch_normalization_80_694265batch_normalization_80_694267*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_69351620
.batch_normalization_80/StatefulPartitionedCall�
 max_pooling2d_34/PartitionedCallPartitionedCall7batch_normalization_80/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_6928172"
 max_pooling2d_34/PartitionedCall�
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0conv2d_70_694271conv2d_70_694273*
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
E__inference_conv2d_70_layer_call_and_return_conditional_losses_6935642#
!conv2d_70/StatefulPartitionedCall�
.batch_normalization_81/StatefulPartitionedCallStatefulPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0batch_normalization_81_694276batch_normalization_81_694278batch_normalization_81_694280batch_normalization_81_694282*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_69361720
.batch_normalization_81/StatefulPartitionedCall�
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_81/StatefulPartitionedCall:output:0conv2d_71_694285conv2d_71_694287*
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
E__inference_conv2d_71_layer_call_and_return_conditional_losses_6936642#
!conv2d_71/StatefulPartitionedCall�
.batch_normalization_82/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0batch_normalization_82_694290batch_normalization_82_694292batch_normalization_82_694294batch_normalization_82_694296*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_69371720
.batch_normalization_82/StatefulPartitionedCall�
 max_pooling2d_35/PartitionedCallPartitionedCall7batch_normalization_82/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_6930372"
 max_pooling2d_35/PartitionedCall�
.batch_normalization_83/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_83_694300batch_normalization_83_694302batch_normalization_83_694304batch_normalization_83_694306*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_69379120
.batch_normalization_83/StatefulPartitionedCall�
flatten_11/PartitionedCallPartitionedCall7batch_normalization_83/StatefulPartitionedCall:output:0*
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
F__inference_flatten_11_layer_call_and_return_conditional_losses_6938332
flatten_11/PartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_11_694310dense_11_694312*
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
D__inference_dense_11_layer_call_and_return_conditional_losses_6938522"
 dense_11/StatefulPartitionedCall�
lambda_11/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
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
E__inference_lambda_11_layer_call_and_return_conditional_losses_6938902
lambda_11/PartitionedCall�
IdentityIdentity"lambda_11/PartitionedCall:output:0/^batch_normalization_77/StatefulPartitionedCall/^batch_normalization_78/StatefulPartitionedCall/^batch_normalization_79/StatefulPartitionedCall/^batch_normalization_80/StatefulPartitionedCall/^batch_normalization_81/StatefulPartitionedCall/^batch_normalization_82/StatefulPartitionedCall/^batch_normalization_83/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_77/StatefulPartitionedCall.batch_normalization_77/StatefulPartitionedCall2`
.batch_normalization_78/StatefulPartitionedCall.batch_normalization_78/StatefulPartitionedCall2`
.batch_normalization_79/StatefulPartitionedCall.batch_normalization_79/StatefulPartitionedCall2`
.batch_normalization_80/StatefulPartitionedCall.batch_normalization_80/StatefulPartitionedCall2`
.batch_normalization_81/StatefulPartitionedCall.batch_normalization_81/StatefulPartitionedCall2`
.batch_normalization_82/StatefulPartitionedCall.batch_normalization_82/StatefulPartitionedCall2`
.batch_normalization_83/StatefulPartitionedCall.batch_normalization_83/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_79_layer_call_fn_695380

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_6926652
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
E__inference_conv2d_68_layer_call_and_return_conditional_losses_695320

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
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_693398

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
M
1__inference_max_pooling2d_33_layer_call_fn_692603

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
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_6925972
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
�
�
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_693773

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
�
�
.__inference_sequential_11_layer_call_fn_694208
conv2d_66_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_11_layer_call_and_return_conditional_losses_6941212
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
_user_specified_nameconv2d_66_input
�
�
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695875

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
7__inference_batch_normalization_83_layer_call_fn_696016

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_6937732
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
�
�
7__inference_batch_normalization_82_layer_call_fn_695888

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_6936992
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

*__inference_conv2d_70_layer_call_fn_695625

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
E__inference_conv2d_70_layer_call_and_return_conditional_losses_6935642
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
E__inference_conv2d_70_layer_call_and_return_conditional_losses_693564

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
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_692885

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
7__inference_batch_normalization_78_layer_call_fn_695245

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_6925802
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
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_693791

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
M
1__inference_max_pooling2d_35_layer_call_fn_693043

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
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_6930372
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
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_695939

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
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_693498

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
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695201

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
�
�
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_693717

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
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695709

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
�
b
F__inference_flatten_11_layer_call_and_return_conditional_losses_693833

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
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_693105

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
�
�
.__inference_sequential_11_layer_call_fn_695013

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
I__inference_sequential_11_layer_call_and_return_conditional_losses_6943172
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
�
F
*__inference_lambda_11_layer_call_fn_696087

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
E__inference_lambda_11_layer_call_and_return_conditional_losses_6938792
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
�

*__inference_conv2d_67_layer_call_fn_695181

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
E__inference_conv2d_67_layer_call_and_return_conditional_losses_6932622
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

a
E__inference_lambda_11_layer_call_and_return_conditional_losses_693890

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
��
�*
!__inference__wrapped_model_692383
conv2d_66_input:
6sequential_11_conv2d_66_conv2d_readvariableop_resource;
7sequential_11_conv2d_66_biasadd_readvariableop_resource@
<sequential_11_batch_normalization_77_readvariableop_resourceB
>sequential_11_batch_normalization_77_readvariableop_1_resourceQ
Msequential_11_batch_normalization_77_fusedbatchnormv3_readvariableop_resourceS
Osequential_11_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resource:
6sequential_11_conv2d_67_conv2d_readvariableop_resource;
7sequential_11_conv2d_67_biasadd_readvariableop_resource@
<sequential_11_batch_normalization_78_readvariableop_resourceB
>sequential_11_batch_normalization_78_readvariableop_1_resourceQ
Msequential_11_batch_normalization_78_fusedbatchnormv3_readvariableop_resourceS
Osequential_11_batch_normalization_78_fusedbatchnormv3_readvariableop_1_resource:
6sequential_11_conv2d_68_conv2d_readvariableop_resource;
7sequential_11_conv2d_68_biasadd_readvariableop_resource@
<sequential_11_batch_normalization_79_readvariableop_resourceB
>sequential_11_batch_normalization_79_readvariableop_1_resourceQ
Msequential_11_batch_normalization_79_fusedbatchnormv3_readvariableop_resourceS
Osequential_11_batch_normalization_79_fusedbatchnormv3_readvariableop_1_resource:
6sequential_11_conv2d_69_conv2d_readvariableop_resource;
7sequential_11_conv2d_69_biasadd_readvariableop_resource@
<sequential_11_batch_normalization_80_readvariableop_resourceB
>sequential_11_batch_normalization_80_readvariableop_1_resourceQ
Msequential_11_batch_normalization_80_fusedbatchnormv3_readvariableop_resourceS
Osequential_11_batch_normalization_80_fusedbatchnormv3_readvariableop_1_resource:
6sequential_11_conv2d_70_conv2d_readvariableop_resource;
7sequential_11_conv2d_70_biasadd_readvariableop_resource@
<sequential_11_batch_normalization_81_readvariableop_resourceB
>sequential_11_batch_normalization_81_readvariableop_1_resourceQ
Msequential_11_batch_normalization_81_fusedbatchnormv3_readvariableop_resourceS
Osequential_11_batch_normalization_81_fusedbatchnormv3_readvariableop_1_resource:
6sequential_11_conv2d_71_conv2d_readvariableop_resource;
7sequential_11_conv2d_71_biasadd_readvariableop_resource@
<sequential_11_batch_normalization_82_readvariableop_resourceB
>sequential_11_batch_normalization_82_readvariableop_1_resourceQ
Msequential_11_batch_normalization_82_fusedbatchnormv3_readvariableop_resourceS
Osequential_11_batch_normalization_82_fusedbatchnormv3_readvariableop_1_resource@
<sequential_11_batch_normalization_83_readvariableop_resourceB
>sequential_11_batch_normalization_83_readvariableop_1_resourceQ
Msequential_11_batch_normalization_83_fusedbatchnormv3_readvariableop_resourceS
Osequential_11_batch_normalization_83_fusedbatchnormv3_readvariableop_1_resource9
5sequential_11_dense_11_matmul_readvariableop_resource:
6sequential_11_dense_11_biasadd_readvariableop_resource
identity��Dsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp�Fsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1�3sequential_11/batch_normalization_77/ReadVariableOp�5sequential_11/batch_normalization_77/ReadVariableOp_1�Dsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp�Fsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1�3sequential_11/batch_normalization_78/ReadVariableOp�5sequential_11/batch_normalization_78/ReadVariableOp_1�Dsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp�Fsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1�3sequential_11/batch_normalization_79/ReadVariableOp�5sequential_11/batch_normalization_79/ReadVariableOp_1�Dsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp�Fsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1�3sequential_11/batch_normalization_80/ReadVariableOp�5sequential_11/batch_normalization_80/ReadVariableOp_1�Dsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp�Fsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1�3sequential_11/batch_normalization_81/ReadVariableOp�5sequential_11/batch_normalization_81/ReadVariableOp_1�Dsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp�Fsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1�3sequential_11/batch_normalization_82/ReadVariableOp�5sequential_11/batch_normalization_82/ReadVariableOp_1�Dsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp�Fsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1�3sequential_11/batch_normalization_83/ReadVariableOp�5sequential_11/batch_normalization_83/ReadVariableOp_1�.sequential_11/conv2d_66/BiasAdd/ReadVariableOp�-sequential_11/conv2d_66/Conv2D/ReadVariableOp�.sequential_11/conv2d_67/BiasAdd/ReadVariableOp�-sequential_11/conv2d_67/Conv2D/ReadVariableOp�.sequential_11/conv2d_68/BiasAdd/ReadVariableOp�-sequential_11/conv2d_68/Conv2D/ReadVariableOp�.sequential_11/conv2d_69/BiasAdd/ReadVariableOp�-sequential_11/conv2d_69/Conv2D/ReadVariableOp�.sequential_11/conv2d_70/BiasAdd/ReadVariableOp�-sequential_11/conv2d_70/Conv2D/ReadVariableOp�.sequential_11/conv2d_71/BiasAdd/ReadVariableOp�-sequential_11/conv2d_71/Conv2D/ReadVariableOp�-sequential_11/dense_11/BiasAdd/ReadVariableOp�,sequential_11/dense_11/MatMul/ReadVariableOp�
-sequential_11/conv2d_66/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_11/conv2d_66/Conv2D/ReadVariableOp�
sequential_11/conv2d_66/Conv2DConv2Dconv2d_66_input5sequential_11/conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2 
sequential_11/conv2d_66/Conv2D�
.sequential_11/conv2d_66/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_66_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_11/conv2d_66/BiasAdd/ReadVariableOp�
sequential_11/conv2d_66/BiasAddBiasAdd'sequential_11/conv2d_66/Conv2D:output:06sequential_11/conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2!
sequential_11/conv2d_66/BiasAdd�
sequential_11/conv2d_66/ReluRelu(sequential_11/conv2d_66/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
sequential_11/conv2d_66/Relu�
3sequential_11/batch_normalization_77/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_77_readvariableop_resource*
_output_shapes
: *
dtype025
3sequential_11/batch_normalization_77/ReadVariableOp�
5sequential_11/batch_normalization_77/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_77_readvariableop_1_resource*
_output_shapes
: *
dtype027
5sequential_11/batch_normalization_77/ReadVariableOp_1�
Dsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp�
Fsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1�
5sequential_11/batch_normalization_77/FusedBatchNormV3FusedBatchNormV3*sequential_11/conv2d_66/Relu:activations:0;sequential_11/batch_normalization_77/ReadVariableOp:value:0=sequential_11/batch_normalization_77/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 27
5sequential_11/batch_normalization_77/FusedBatchNormV3�
-sequential_11/conv2d_67/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_67_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02/
-sequential_11/conv2d_67/Conv2D/ReadVariableOp�
sequential_11/conv2d_67/Conv2DConv2D9sequential_11/batch_normalization_77/FusedBatchNormV3:y:05sequential_11/conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2 
sequential_11/conv2d_67/Conv2D�
.sequential_11/conv2d_67/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_67_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_11/conv2d_67/BiasAdd/ReadVariableOp�
sequential_11/conv2d_67/BiasAddBiasAdd'sequential_11/conv2d_67/Conv2D:output:06sequential_11/conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2!
sequential_11/conv2d_67/BiasAdd�
sequential_11/conv2d_67/ReluRelu(sequential_11/conv2d_67/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
sequential_11/conv2d_67/Relu�
3sequential_11/batch_normalization_78/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_78_readvariableop_resource*
_output_shapes
: *
dtype025
3sequential_11/batch_normalization_78/ReadVariableOp�
5sequential_11/batch_normalization_78/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_78_readvariableop_1_resource*
_output_shapes
: *
dtype027
5sequential_11/batch_normalization_78/ReadVariableOp_1�
Dsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_78_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp�
Fsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_78_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1�
5sequential_11/batch_normalization_78/FusedBatchNormV3FusedBatchNormV3*sequential_11/conv2d_67/Relu:activations:0;sequential_11/batch_normalization_78/ReadVariableOp:value:0=sequential_11/batch_normalization_78/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 27
5sequential_11/batch_normalization_78/FusedBatchNormV3�
&sequential_11/max_pooling2d_33/MaxPoolMaxPool9sequential_11/batch_normalization_78/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2(
&sequential_11/max_pooling2d_33/MaxPool�
-sequential_11/conv2d_68/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_68_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_11/conv2d_68/Conv2D/ReadVariableOp�
sequential_11/conv2d_68/Conv2DConv2D/sequential_11/max_pooling2d_33/MaxPool:output:05sequential_11/conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2 
sequential_11/conv2d_68/Conv2D�
.sequential_11/conv2d_68/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_68_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_11/conv2d_68/BiasAdd/ReadVariableOp�
sequential_11/conv2d_68/BiasAddBiasAdd'sequential_11/conv2d_68/Conv2D:output:06sequential_11/conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2!
sequential_11/conv2d_68/BiasAdd�
sequential_11/conv2d_68/ReluRelu(sequential_11/conv2d_68/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential_11/conv2d_68/Relu�
3sequential_11/batch_normalization_79/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_79_readvariableop_resource*
_output_shapes
:@*
dtype025
3sequential_11/batch_normalization_79/ReadVariableOp�
5sequential_11/batch_normalization_79/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_79_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5sequential_11/batch_normalization_79/ReadVariableOp_1�
Dsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_79_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp�
Fsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_79_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1�
5sequential_11/batch_normalization_79/FusedBatchNormV3FusedBatchNormV3*sequential_11/conv2d_68/Relu:activations:0;sequential_11/batch_normalization_79/ReadVariableOp:value:0=sequential_11/batch_normalization_79/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5sequential_11/batch_normalization_79/FusedBatchNormV3�
-sequential_11/conv2d_69/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_69_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_11/conv2d_69/Conv2D/ReadVariableOp�
sequential_11/conv2d_69/Conv2DConv2D9sequential_11/batch_normalization_79/FusedBatchNormV3:y:05sequential_11/conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2 
sequential_11/conv2d_69/Conv2D�
.sequential_11/conv2d_69/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_69_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_11/conv2d_69/BiasAdd/ReadVariableOp�
sequential_11/conv2d_69/BiasAddBiasAdd'sequential_11/conv2d_69/Conv2D:output:06sequential_11/conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2!
sequential_11/conv2d_69/BiasAdd�
sequential_11/conv2d_69/ReluRelu(sequential_11/conv2d_69/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential_11/conv2d_69/Relu�
3sequential_11/batch_normalization_80/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_80_readvariableop_resource*
_output_shapes
:@*
dtype025
3sequential_11/batch_normalization_80/ReadVariableOp�
5sequential_11/batch_normalization_80/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_80_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5sequential_11/batch_normalization_80/ReadVariableOp_1�
Dsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_80_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp�
Fsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_80_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1�
5sequential_11/batch_normalization_80/FusedBatchNormV3FusedBatchNormV3*sequential_11/conv2d_69/Relu:activations:0;sequential_11/batch_normalization_80/ReadVariableOp:value:0=sequential_11/batch_normalization_80/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5sequential_11/batch_normalization_80/FusedBatchNormV3�
&sequential_11/max_pooling2d_34/MaxPoolMaxPool9sequential_11/batch_normalization_80/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2(
&sequential_11/max_pooling2d_34/MaxPool�
-sequential_11/conv2d_70/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_70_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02/
-sequential_11/conv2d_70/Conv2D/ReadVariableOp�
sequential_11/conv2d_70/Conv2DConv2D/sequential_11/max_pooling2d_34/MaxPool:output:05sequential_11/conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2 
sequential_11/conv2d_70/Conv2D�
.sequential_11/conv2d_70/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_70_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.sequential_11/conv2d_70/BiasAdd/ReadVariableOp�
sequential_11/conv2d_70/BiasAddBiasAdd'sequential_11/conv2d_70/Conv2D:output:06sequential_11/conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2!
sequential_11/conv2d_70/BiasAdd�
sequential_11/conv2d_70/ReluRelu(sequential_11/conv2d_70/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_11/conv2d_70/Relu�
3sequential_11/batch_normalization_81/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_81_readvariableop_resource*
_output_shapes	
:�*
dtype025
3sequential_11/batch_normalization_81/ReadVariableOp�
5sequential_11/batch_normalization_81/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_81_readvariableop_1_resource*
_output_shapes	
:�*
dtype027
5sequential_11/batch_normalization_81/ReadVariableOp_1�
Dsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_81_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp�
Fsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_81_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02H
Fsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1�
5sequential_11/batch_normalization_81/FusedBatchNormV3FusedBatchNormV3*sequential_11/conv2d_70/Relu:activations:0;sequential_11/batch_normalization_81/ReadVariableOp:value:0=sequential_11/batch_normalization_81/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 27
5sequential_11/batch_normalization_81/FusedBatchNormV3�
-sequential_11/conv2d_71/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_71_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02/
-sequential_11/conv2d_71/Conv2D/ReadVariableOp�
sequential_11/conv2d_71/Conv2DConv2D9sequential_11/batch_normalization_81/FusedBatchNormV3:y:05sequential_11/conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2 
sequential_11/conv2d_71/Conv2D�
.sequential_11/conv2d_71/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_71_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.sequential_11/conv2d_71/BiasAdd/ReadVariableOp�
sequential_11/conv2d_71/BiasAddBiasAdd'sequential_11/conv2d_71/Conv2D:output:06sequential_11/conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2!
sequential_11/conv2d_71/BiasAdd�
sequential_11/conv2d_71/ReluRelu(sequential_11/conv2d_71/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_11/conv2d_71/Relu�
3sequential_11/batch_normalization_82/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_82_readvariableop_resource*
_output_shapes	
:�*
dtype025
3sequential_11/batch_normalization_82/ReadVariableOp�
5sequential_11/batch_normalization_82/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_82_readvariableop_1_resource*
_output_shapes	
:�*
dtype027
5sequential_11/batch_normalization_82/ReadVariableOp_1�
Dsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_82_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp�
Fsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_82_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02H
Fsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1�
5sequential_11/batch_normalization_82/FusedBatchNormV3FusedBatchNormV3*sequential_11/conv2d_71/Relu:activations:0;sequential_11/batch_normalization_82/ReadVariableOp:value:0=sequential_11/batch_normalization_82/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 27
5sequential_11/batch_normalization_82/FusedBatchNormV3�
&sequential_11/max_pooling2d_35/MaxPoolMaxPool9sequential_11/batch_normalization_82/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2(
&sequential_11/max_pooling2d_35/MaxPool�
3sequential_11/batch_normalization_83/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_83_readvariableop_resource*
_output_shapes	
:�*
dtype025
3sequential_11/batch_normalization_83/ReadVariableOp�
5sequential_11/batch_normalization_83/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_83_readvariableop_1_resource*
_output_shapes	
:�*
dtype027
5sequential_11/batch_normalization_83/ReadVariableOp_1�
Dsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_83_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp�
Fsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_83_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02H
Fsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1�
5sequential_11/batch_normalization_83/FusedBatchNormV3FusedBatchNormV3/sequential_11/max_pooling2d_35/MaxPool:output:0;sequential_11/batch_normalization_83/ReadVariableOp:value:0=sequential_11/batch_normalization_83/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 27
5sequential_11/batch_normalization_83/FusedBatchNormV3�
sequential_11/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2 
sequential_11/flatten_11/Const�
 sequential_11/flatten_11/ReshapeReshape9sequential_11/batch_normalization_83/FusedBatchNormV3:y:0'sequential_11/flatten_11/Const:output:0*
T0*(
_output_shapes
:����������2"
 sequential_11/flatten_11/Reshape�
,sequential_11/dense_11/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,sequential_11/dense_11/MatMul/ReadVariableOp�
sequential_11/dense_11/MatMulMatMul)sequential_11/flatten_11/Reshape:output:04sequential_11/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_11/dense_11/MatMul�
-sequential_11/dense_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_11/dense_11/BiasAdd/ReadVariableOp�
sequential_11/dense_11/BiasAddBiasAdd'sequential_11/dense_11/MatMul:product:05sequential_11/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_11/dense_11/BiasAdd�
sequential_11/dense_11/SigmoidSigmoid'sequential_11/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:����������2 
sequential_11/dense_11/Sigmoid�
+sequential_11/lambda_11/l2_normalize/SquareSquare"sequential_11/dense_11/Sigmoid:y:0*
T0*(
_output_shapes
:����������2-
+sequential_11/lambda_11/l2_normalize/Square�
:sequential_11/lambda_11/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_11/lambda_11/l2_normalize/Sum/reduction_indices�
(sequential_11/lambda_11/l2_normalize/SumSum/sequential_11/lambda_11/l2_normalize/Square:y:0Csequential_11/lambda_11/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2*
(sequential_11/lambda_11/l2_normalize/Sum�
.sequential_11/lambda_11/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+20
.sequential_11/lambda_11/l2_normalize/Maximum/y�
,sequential_11/lambda_11/l2_normalize/MaximumMaximum1sequential_11/lambda_11/l2_normalize/Sum:output:07sequential_11/lambda_11/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2.
,sequential_11/lambda_11/l2_normalize/Maximum�
*sequential_11/lambda_11/l2_normalize/RsqrtRsqrt0sequential_11/lambda_11/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2,
*sequential_11/lambda_11/l2_normalize/Rsqrt�
$sequential_11/lambda_11/l2_normalizeMul"sequential_11/dense_11/Sigmoid:y:0.sequential_11/lambda_11/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2&
$sequential_11/lambda_11/l2_normalize�
IdentityIdentity(sequential_11/lambda_11/l2_normalize:z:0E^sequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_77/ReadVariableOp6^sequential_11/batch_normalization_77/ReadVariableOp_1E^sequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_78/ReadVariableOp6^sequential_11/batch_normalization_78/ReadVariableOp_1E^sequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_79/ReadVariableOp6^sequential_11/batch_normalization_79/ReadVariableOp_1E^sequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_80/ReadVariableOp6^sequential_11/batch_normalization_80/ReadVariableOp_1E^sequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_81/ReadVariableOp6^sequential_11/batch_normalization_81/ReadVariableOp_1E^sequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_82/ReadVariableOp6^sequential_11/batch_normalization_82/ReadVariableOp_1E^sequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_83/ReadVariableOp6^sequential_11/batch_normalization_83/ReadVariableOp_1/^sequential_11/conv2d_66/BiasAdd/ReadVariableOp.^sequential_11/conv2d_66/Conv2D/ReadVariableOp/^sequential_11/conv2d_67/BiasAdd/ReadVariableOp.^sequential_11/conv2d_67/Conv2D/ReadVariableOp/^sequential_11/conv2d_68/BiasAdd/ReadVariableOp.^sequential_11/conv2d_68/Conv2D/ReadVariableOp/^sequential_11/conv2d_69/BiasAdd/ReadVariableOp.^sequential_11/conv2d_69/Conv2D/ReadVariableOp/^sequential_11/conv2d_70/BiasAdd/ReadVariableOp.^sequential_11/conv2d_70/Conv2D/ReadVariableOp/^sequential_11/conv2d_71/BiasAdd/ReadVariableOp.^sequential_11/conv2d_71/Conv2D/ReadVariableOp.^sequential_11/dense_11/BiasAdd/ReadVariableOp-^sequential_11/dense_11/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2�
Dsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp2�
Fsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_77/ReadVariableOp3sequential_11/batch_normalization_77/ReadVariableOp2n
5sequential_11/batch_normalization_77/ReadVariableOp_15sequential_11/batch_normalization_77/ReadVariableOp_12�
Dsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp2�
Fsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_78/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_78/ReadVariableOp3sequential_11/batch_normalization_78/ReadVariableOp2n
5sequential_11/batch_normalization_78/ReadVariableOp_15sequential_11/batch_normalization_78/ReadVariableOp_12�
Dsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp2�
Fsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_79/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_79/ReadVariableOp3sequential_11/batch_normalization_79/ReadVariableOp2n
5sequential_11/batch_normalization_79/ReadVariableOp_15sequential_11/batch_normalization_79/ReadVariableOp_12�
Dsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp2�
Fsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_80/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_80/ReadVariableOp3sequential_11/batch_normalization_80/ReadVariableOp2n
5sequential_11/batch_normalization_80/ReadVariableOp_15sequential_11/batch_normalization_80/ReadVariableOp_12�
Dsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp2�
Fsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_81/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_81/ReadVariableOp3sequential_11/batch_normalization_81/ReadVariableOp2n
5sequential_11/batch_normalization_81/ReadVariableOp_15sequential_11/batch_normalization_81/ReadVariableOp_12�
Dsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp2�
Fsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_82/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_82/ReadVariableOp3sequential_11/batch_normalization_82/ReadVariableOp2n
5sequential_11/batch_normalization_82/ReadVariableOp_15sequential_11/batch_normalization_82/ReadVariableOp_12�
Dsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp2�
Fsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_83/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_83/ReadVariableOp3sequential_11/batch_normalization_83/ReadVariableOp2n
5sequential_11/batch_normalization_83/ReadVariableOp_15sequential_11/batch_normalization_83/ReadVariableOp_12`
.sequential_11/conv2d_66/BiasAdd/ReadVariableOp.sequential_11/conv2d_66/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_66/Conv2D/ReadVariableOp-sequential_11/conv2d_66/Conv2D/ReadVariableOp2`
.sequential_11/conv2d_67/BiasAdd/ReadVariableOp.sequential_11/conv2d_67/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_67/Conv2D/ReadVariableOp-sequential_11/conv2d_67/Conv2D/ReadVariableOp2`
.sequential_11/conv2d_68/BiasAdd/ReadVariableOp.sequential_11/conv2d_68/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_68/Conv2D/ReadVariableOp-sequential_11/conv2d_68/Conv2D/ReadVariableOp2`
.sequential_11/conv2d_69/BiasAdd/ReadVariableOp.sequential_11/conv2d_69/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_69/Conv2D/ReadVariableOp-sequential_11/conv2d_69/Conv2D/ReadVariableOp2`
.sequential_11/conv2d_70/BiasAdd/ReadVariableOp.sequential_11/conv2d_70/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_70/Conv2D/ReadVariableOp-sequential_11/conv2d_70/Conv2D/ReadVariableOp2`
.sequential_11/conv2d_71/BiasAdd/ReadVariableOp.sequential_11/conv2d_71/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_71/Conv2D/ReadVariableOp-sequential_11/conv2d_71/Conv2D/ReadVariableOp2^
-sequential_11/dense_11/BiasAdd/ReadVariableOp-sequential_11/dense_11/BiasAdd/ReadVariableOp2\
,sequential_11/dense_11/MatMul/ReadVariableOp,sequential_11/dense_11/MatMul/ReadVariableOp:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_66_input
�
�
7__inference_batch_normalization_81_layer_call_fn_695689

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_6929162
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
�
�
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_693617

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
�
�
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_693516

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
�i
�
I__inference_sequential_11_layer_call_and_return_conditional_losses_694121

inputs
conv2d_66_694017
conv2d_66_694019!
batch_normalization_77_694022!
batch_normalization_77_694024!
batch_normalization_77_694026!
batch_normalization_77_694028
conv2d_67_694031
conv2d_67_694033!
batch_normalization_78_694036!
batch_normalization_78_694038!
batch_normalization_78_694040!
batch_normalization_78_694042
conv2d_68_694046
conv2d_68_694048!
batch_normalization_79_694051!
batch_normalization_79_694053!
batch_normalization_79_694055!
batch_normalization_79_694057
conv2d_69_694060
conv2d_69_694062!
batch_normalization_80_694065!
batch_normalization_80_694067!
batch_normalization_80_694069!
batch_normalization_80_694071
conv2d_70_694075
conv2d_70_694077!
batch_normalization_81_694080!
batch_normalization_81_694082!
batch_normalization_81_694084!
batch_normalization_81_694086
conv2d_71_694089
conv2d_71_694091!
batch_normalization_82_694094!
batch_normalization_82_694096!
batch_normalization_82_694098!
batch_normalization_82_694100!
batch_normalization_83_694104!
batch_normalization_83_694106!
batch_normalization_83_694108!
batch_normalization_83_694110
dense_11_694114
dense_11_694116
identity��.batch_normalization_77/StatefulPartitionedCall�.batch_normalization_78/StatefulPartitionedCall�.batch_normalization_79/StatefulPartitionedCall�.batch_normalization_80/StatefulPartitionedCall�.batch_normalization_81/StatefulPartitionedCall�.batch_normalization_82/StatefulPartitionedCall�.batch_normalization_83/StatefulPartitionedCall�!conv2d_66/StatefulPartitionedCall�!conv2d_67/StatefulPartitionedCall�!conv2d_68/StatefulPartitionedCall�!conv2d_69/StatefulPartitionedCall�!conv2d_70/StatefulPartitionedCall�!conv2d_71/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_66_694017conv2d_66_694019*
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
E__inference_conv2d_66_layer_call_and_return_conditional_losses_6931622#
!conv2d_66/StatefulPartitionedCall�
.batch_normalization_77/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_77_694022batch_normalization_77_694024batch_normalization_77_694026batch_normalization_77_694028*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_69319720
.batch_normalization_77/StatefulPartitionedCall�
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_77/StatefulPartitionedCall:output:0conv2d_67_694031conv2d_67_694033*
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
E__inference_conv2d_67_layer_call_and_return_conditional_losses_6932622#
!conv2d_67/StatefulPartitionedCall�
.batch_normalization_78/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_78_694036batch_normalization_78_694038batch_normalization_78_694040batch_normalization_78_694042*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_69329720
.batch_normalization_78/StatefulPartitionedCall�
 max_pooling2d_33/PartitionedCallPartitionedCall7batch_normalization_78/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_6925972"
 max_pooling2d_33/PartitionedCall�
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0conv2d_68_694046conv2d_68_694048*
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
E__inference_conv2d_68_layer_call_and_return_conditional_losses_6933632#
!conv2d_68/StatefulPartitionedCall�
.batch_normalization_79/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_79_694051batch_normalization_79_694053batch_normalization_79_694055batch_normalization_79_694057*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_69339820
.batch_normalization_79/StatefulPartitionedCall�
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_79/StatefulPartitionedCall:output:0conv2d_69_694060conv2d_69_694062*
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
E__inference_conv2d_69_layer_call_and_return_conditional_losses_6934632#
!conv2d_69/StatefulPartitionedCall�
.batch_normalization_80/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0batch_normalization_80_694065batch_normalization_80_694067batch_normalization_80_694069batch_normalization_80_694071*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_69349820
.batch_normalization_80/StatefulPartitionedCall�
 max_pooling2d_34/PartitionedCallPartitionedCall7batch_normalization_80/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_6928172"
 max_pooling2d_34/PartitionedCall�
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0conv2d_70_694075conv2d_70_694077*
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
E__inference_conv2d_70_layer_call_and_return_conditional_losses_6935642#
!conv2d_70/StatefulPartitionedCall�
.batch_normalization_81/StatefulPartitionedCallStatefulPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0batch_normalization_81_694080batch_normalization_81_694082batch_normalization_81_694084batch_normalization_81_694086*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_69359920
.batch_normalization_81/StatefulPartitionedCall�
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_81/StatefulPartitionedCall:output:0conv2d_71_694089conv2d_71_694091*
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
E__inference_conv2d_71_layer_call_and_return_conditional_losses_6936642#
!conv2d_71/StatefulPartitionedCall�
.batch_normalization_82/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0batch_normalization_82_694094batch_normalization_82_694096batch_normalization_82_694098batch_normalization_82_694100*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_69369920
.batch_normalization_82/StatefulPartitionedCall�
 max_pooling2d_35/PartitionedCallPartitionedCall7batch_normalization_82/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_6930372"
 max_pooling2d_35/PartitionedCall�
.batch_normalization_83/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_83_694104batch_normalization_83_694106batch_normalization_83_694108batch_normalization_83_694110*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_69377320
.batch_normalization_83/StatefulPartitionedCall�
flatten_11/PartitionedCallPartitionedCall7batch_normalization_83/StatefulPartitionedCall:output:0*
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
F__inference_flatten_11_layer_call_and_return_conditional_losses_6938332
flatten_11/PartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_11_694114dense_11_694116*
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
D__inference_dense_11_layer_call_and_return_conditional_losses_6938522"
 dense_11/StatefulPartitionedCall�
lambda_11/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
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
E__inference_lambda_11_layer_call_and_return_conditional_losses_6938792
lambda_11/PartitionedCall�
IdentityIdentity"lambda_11/PartitionedCall:output:0/^batch_normalization_77/StatefulPartitionedCall/^batch_normalization_78/StatefulPartitionedCall/^batch_normalization_79/StatefulPartitionedCall/^batch_normalization_80/StatefulPartitionedCall/^batch_normalization_81/StatefulPartitionedCall/^batch_normalization_82/StatefulPartitionedCall/^batch_normalization_83/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_77/StatefulPartitionedCall.batch_normalization_77/StatefulPartitionedCall2`
.batch_normalization_78/StatefulPartitionedCall.batch_normalization_78/StatefulPartitionedCall2`
.batch_normalization_79/StatefulPartitionedCall.batch_normalization_79/StatefulPartitionedCall2`
.batch_normalization_80/StatefulPartitionedCall.batch_normalization_80/StatefulPartitionedCall2`
.batch_normalization_81/StatefulPartitionedCall.batch_normalization_81/StatefulPartitionedCall2`
.batch_normalization_82/StatefulPartitionedCall.batch_normalization_82/StatefulPartitionedCall2`
.batch_normalization_83/StatefulPartitionedCall.batch_normalization_83/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_80_layer_call_fn_695605

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_6935162
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
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_695985

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
�
�
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_692696

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
�
~
)__inference_dense_11_layer_call_fn_696060

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
D__inference_dense_11_layer_call_and_return_conditional_losses_6938522
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
�
�
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695349

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
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_692665

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
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695793

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
7__inference_batch_normalization_83_layer_call_fn_695952

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_6931052
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
�

�
E__inference_conv2d_68_layer_call_and_return_conditional_losses_693363

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
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695727

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
�

a
E__inference_lambda_11_layer_call_and_return_conditional_losses_693879

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
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695135

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
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695857

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
E__inference_conv2d_71_layer_call_and_return_conditional_losses_693664

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
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_692476

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
�

*__inference_conv2d_69_layer_call_fn_695477

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
E__inference_conv2d_69_layer_call_and_return_conditional_losses_6934632
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
�
�
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_692580

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
7__inference_batch_normalization_77_layer_call_fn_695148

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_6924452
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
E__inference_conv2d_67_layer_call_and_return_conditional_losses_693262

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
�
�
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695117

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
�
�
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695663

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
7__inference_batch_normalization_79_layer_call_fn_695444

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_6933982
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
�
G
+__inference_flatten_11_layer_call_fn_696040

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
F__inference_flatten_11_layer_call_and_return_conditional_losses_6938332
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
�
�
7__inference_batch_normalization_80_layer_call_fn_695541

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_6928002
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
�
�
.__inference_sequential_11_layer_call_fn_694404
conv2d_66_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_11_layer_call_and_return_conditional_losses_6943172
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
_user_specified_nameconv2d_66_input
�
�
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_692916

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
�
�
7__inference_batch_normalization_82_layer_call_fn_695824

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_6929892
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
�

�
E__inference_conv2d_69_layer_call_and_return_conditional_losses_693463

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
�

*__inference_conv2d_66_layer_call_fn_695033

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
E__inference_conv2d_66_layer_call_and_return_conditional_losses_6931622
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
�
�
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695053

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
�
�
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695413

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
�

a
E__inference_lambda_11_layer_call_and_return_conditional_losses_696082

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
�

�
E__inference_conv2d_66_layer_call_and_return_conditional_losses_693162

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
��
�!
I__inference_sequential_11_layer_call_and_return_conditional_losses_694835

inputs,
(conv2d_66_conv2d_readvariableop_resource-
)conv2d_66_biasadd_readvariableop_resource2
.batch_normalization_77_readvariableop_resource4
0batch_normalization_77_readvariableop_1_resourceC
?batch_normalization_77_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_67_conv2d_readvariableop_resource-
)conv2d_67_biasadd_readvariableop_resource2
.batch_normalization_78_readvariableop_resource4
0batch_normalization_78_readvariableop_1_resourceC
?batch_normalization_78_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_78_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_68_conv2d_readvariableop_resource-
)conv2d_68_biasadd_readvariableop_resource2
.batch_normalization_79_readvariableop_resource4
0batch_normalization_79_readvariableop_1_resourceC
?batch_normalization_79_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_79_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_69_conv2d_readvariableop_resource-
)conv2d_69_biasadd_readvariableop_resource2
.batch_normalization_80_readvariableop_resource4
0batch_normalization_80_readvariableop_1_resourceC
?batch_normalization_80_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_80_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_70_conv2d_readvariableop_resource-
)conv2d_70_biasadd_readvariableop_resource2
.batch_normalization_81_readvariableop_resource4
0batch_normalization_81_readvariableop_1_resourceC
?batch_normalization_81_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_81_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_71_conv2d_readvariableop_resource-
)conv2d_71_biasadd_readvariableop_resource2
.batch_normalization_82_readvariableop_resource4
0batch_normalization_82_readvariableop_1_resourceC
?batch_normalization_82_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_82_fusedbatchnormv3_readvariableop_1_resource2
.batch_normalization_83_readvariableop_resource4
0batch_normalization_83_readvariableop_1_resourceC
?batch_normalization_83_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_83_fusedbatchnormv3_readvariableop_1_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity��6batch_normalization_77/FusedBatchNormV3/ReadVariableOp�8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_77/ReadVariableOp�'batch_normalization_77/ReadVariableOp_1�6batch_normalization_78/FusedBatchNormV3/ReadVariableOp�8batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_78/ReadVariableOp�'batch_normalization_78/ReadVariableOp_1�6batch_normalization_79/FusedBatchNormV3/ReadVariableOp�8batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_79/ReadVariableOp�'batch_normalization_79/ReadVariableOp_1�6batch_normalization_80/FusedBatchNormV3/ReadVariableOp�8batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_80/ReadVariableOp�'batch_normalization_80/ReadVariableOp_1�6batch_normalization_81/FusedBatchNormV3/ReadVariableOp�8batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_81/ReadVariableOp�'batch_normalization_81/ReadVariableOp_1�6batch_normalization_82/FusedBatchNormV3/ReadVariableOp�8batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_82/ReadVariableOp�'batch_normalization_82/ReadVariableOp_1�6batch_normalization_83/FusedBatchNormV3/ReadVariableOp�8batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_83/ReadVariableOp�'batch_normalization_83/ReadVariableOp_1� conv2d_66/BiasAdd/ReadVariableOp�conv2d_66/Conv2D/ReadVariableOp� conv2d_67/BiasAdd/ReadVariableOp�conv2d_67/Conv2D/ReadVariableOp� conv2d_68/BiasAdd/ReadVariableOp�conv2d_68/Conv2D/ReadVariableOp� conv2d_69/BiasAdd/ReadVariableOp�conv2d_69/Conv2D/ReadVariableOp� conv2d_70/BiasAdd/ReadVariableOp�conv2d_70/Conv2D/ReadVariableOp� conv2d_71/BiasAdd/ReadVariableOp�conv2d_71/Conv2D/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_66/Conv2D/ReadVariableOp�
conv2d_66/Conv2DConv2Dinputs'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_66/Conv2D�
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_66/BiasAdd/ReadVariableOp�
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_66/BiasAdd~
conv2d_66/ReluReluconv2d_66/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_66/Relu�
%batch_normalization_77/ReadVariableOpReadVariableOp.batch_normalization_77_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_77/ReadVariableOp�
'batch_normalization_77/ReadVariableOp_1ReadVariableOp0batch_normalization_77_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_77/ReadVariableOp_1�
6batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_77/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_77/FusedBatchNormV3FusedBatchNormV3conv2d_66/Relu:activations:0-batch_normalization_77/ReadVariableOp:value:0/batch_normalization_77/ReadVariableOp_1:value:0>batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_77/FusedBatchNormV3�
conv2d_67/Conv2D/ReadVariableOpReadVariableOp(conv2d_67_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_67/Conv2D/ReadVariableOp�
conv2d_67/Conv2DConv2D+batch_normalization_77/FusedBatchNormV3:y:0'conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_67/Conv2D�
 conv2d_67/BiasAdd/ReadVariableOpReadVariableOp)conv2d_67_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_67/BiasAdd/ReadVariableOp�
conv2d_67/BiasAddBiasAddconv2d_67/Conv2D:output:0(conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_67/BiasAdd~
conv2d_67/ReluReluconv2d_67/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_67/Relu�
%batch_normalization_78/ReadVariableOpReadVariableOp.batch_normalization_78_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_78/ReadVariableOp�
'batch_normalization_78/ReadVariableOp_1ReadVariableOp0batch_normalization_78_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_78/ReadVariableOp_1�
6batch_normalization_78/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_78_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_78/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_78_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_78/FusedBatchNormV3FusedBatchNormV3conv2d_67/Relu:activations:0-batch_normalization_78/ReadVariableOp:value:0/batch_normalization_78/ReadVariableOp_1:value:0>batch_normalization_78/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_78/FusedBatchNormV3�
max_pooling2d_33/MaxPoolMaxPool+batch_normalization_78/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_33/MaxPool�
conv2d_68/Conv2D/ReadVariableOpReadVariableOp(conv2d_68_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_68/Conv2D/ReadVariableOp�
conv2d_68/Conv2DConv2D!max_pooling2d_33/MaxPool:output:0'conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_68/Conv2D�
 conv2d_68/BiasAdd/ReadVariableOpReadVariableOp)conv2d_68_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_68/BiasAdd/ReadVariableOp�
conv2d_68/BiasAddBiasAddconv2d_68/Conv2D:output:0(conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_68/BiasAdd~
conv2d_68/ReluReluconv2d_68/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_68/Relu�
%batch_normalization_79/ReadVariableOpReadVariableOp.batch_normalization_79_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_79/ReadVariableOp�
'batch_normalization_79/ReadVariableOp_1ReadVariableOp0batch_normalization_79_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_79/ReadVariableOp_1�
6batch_normalization_79/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_79_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_79/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_79_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_79/FusedBatchNormV3FusedBatchNormV3conv2d_68/Relu:activations:0-batch_normalization_79/ReadVariableOp:value:0/batch_normalization_79/ReadVariableOp_1:value:0>batch_normalization_79/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_79/FusedBatchNormV3�
conv2d_69/Conv2D/ReadVariableOpReadVariableOp(conv2d_69_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_69/Conv2D/ReadVariableOp�
conv2d_69/Conv2DConv2D+batch_normalization_79/FusedBatchNormV3:y:0'conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_69/Conv2D�
 conv2d_69/BiasAdd/ReadVariableOpReadVariableOp)conv2d_69_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_69/BiasAdd/ReadVariableOp�
conv2d_69/BiasAddBiasAddconv2d_69/Conv2D:output:0(conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_69/BiasAdd~
conv2d_69/ReluReluconv2d_69/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_69/Relu�
%batch_normalization_80/ReadVariableOpReadVariableOp.batch_normalization_80_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_80/ReadVariableOp�
'batch_normalization_80/ReadVariableOp_1ReadVariableOp0batch_normalization_80_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_80/ReadVariableOp_1�
6batch_normalization_80/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_80_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_80/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_80_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_80/FusedBatchNormV3FusedBatchNormV3conv2d_69/Relu:activations:0-batch_normalization_80/ReadVariableOp:value:0/batch_normalization_80/ReadVariableOp_1:value:0>batch_normalization_80/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_80/FusedBatchNormV3�
max_pooling2d_34/MaxPoolMaxPool+batch_normalization_80/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_34/MaxPool�
conv2d_70/Conv2D/ReadVariableOpReadVariableOp(conv2d_70_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_70/Conv2D/ReadVariableOp�
conv2d_70/Conv2DConv2D!max_pooling2d_34/MaxPool:output:0'conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_70/Conv2D�
 conv2d_70/BiasAdd/ReadVariableOpReadVariableOp)conv2d_70_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_70/BiasAdd/ReadVariableOp�
conv2d_70/BiasAddBiasAddconv2d_70/Conv2D:output:0(conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_70/BiasAdd
conv2d_70/ReluReluconv2d_70/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_70/Relu�
%batch_normalization_81/ReadVariableOpReadVariableOp.batch_normalization_81_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_81/ReadVariableOp�
'batch_normalization_81/ReadVariableOp_1ReadVariableOp0batch_normalization_81_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_81/ReadVariableOp_1�
6batch_normalization_81/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_81_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_81/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_81_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_81/FusedBatchNormV3FusedBatchNormV3conv2d_70/Relu:activations:0-batch_normalization_81/ReadVariableOp:value:0/batch_normalization_81/ReadVariableOp_1:value:0>batch_normalization_81/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_81/FusedBatchNormV3�
conv2d_71/Conv2D/ReadVariableOpReadVariableOp(conv2d_71_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_71/Conv2D/ReadVariableOp�
conv2d_71/Conv2DConv2D+batch_normalization_81/FusedBatchNormV3:y:0'conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_71/Conv2D�
 conv2d_71/BiasAdd/ReadVariableOpReadVariableOp)conv2d_71_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_71/BiasAdd/ReadVariableOp�
conv2d_71/BiasAddBiasAddconv2d_71/Conv2D:output:0(conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_71/BiasAdd
conv2d_71/ReluReluconv2d_71/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_71/Relu�
%batch_normalization_82/ReadVariableOpReadVariableOp.batch_normalization_82_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_82/ReadVariableOp�
'batch_normalization_82/ReadVariableOp_1ReadVariableOp0batch_normalization_82_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_82/ReadVariableOp_1�
6batch_normalization_82/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_82_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_82/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_82_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_82/FusedBatchNormV3FusedBatchNormV3conv2d_71/Relu:activations:0-batch_normalization_82/ReadVariableOp:value:0/batch_normalization_82/ReadVariableOp_1:value:0>batch_normalization_82/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_82/FusedBatchNormV3�
max_pooling2d_35/MaxPoolMaxPool+batch_normalization_82/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_35/MaxPool�
%batch_normalization_83/ReadVariableOpReadVariableOp.batch_normalization_83_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_83/ReadVariableOp�
'batch_normalization_83/ReadVariableOp_1ReadVariableOp0batch_normalization_83_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_83/ReadVariableOp_1�
6batch_normalization_83/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_83_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_83/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_83_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_83/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_35/MaxPool:output:0-batch_normalization_83/ReadVariableOp:value:0/batch_normalization_83/ReadVariableOp_1:value:0>batch_normalization_83/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_83/FusedBatchNormV3u
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_11/Const�
flatten_11/ReshapeReshape+batch_normalization_83/FusedBatchNormV3:y:0flatten_11/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_11/Reshape�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMulflatten_11/Reshape:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_11/BiasAdd}
dense_11/SigmoidSigmoiddense_11/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_11/Sigmoid�
lambda_11/l2_normalize/SquareSquaredense_11/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
lambda_11/l2_normalize/Square�
,lambda_11/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,lambda_11/l2_normalize/Sum/reduction_indices�
lambda_11/l2_normalize/SumSum!lambda_11/l2_normalize/Square:y:05lambda_11/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
lambda_11/l2_normalize/Sum�
 lambda_11/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2"
 lambda_11/l2_normalize/Maximum/y�
lambda_11/l2_normalize/MaximumMaximum#lambda_11/l2_normalize/Sum:output:0)lambda_11/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2 
lambda_11/l2_normalize/Maximum�
lambda_11/l2_normalize/RsqrtRsqrt"lambda_11/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
lambda_11/l2_normalize/Rsqrt�
lambda_11/l2_normalizeMuldense_11/Sigmoid:y:0 lambda_11/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
lambda_11/l2_normalize�
IdentityIdentitylambda_11/l2_normalize:z:07^batch_normalization_77/FusedBatchNormV3/ReadVariableOp9^batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_77/ReadVariableOp(^batch_normalization_77/ReadVariableOp_17^batch_normalization_78/FusedBatchNormV3/ReadVariableOp9^batch_normalization_78/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_78/ReadVariableOp(^batch_normalization_78/ReadVariableOp_17^batch_normalization_79/FusedBatchNormV3/ReadVariableOp9^batch_normalization_79/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_79/ReadVariableOp(^batch_normalization_79/ReadVariableOp_17^batch_normalization_80/FusedBatchNormV3/ReadVariableOp9^batch_normalization_80/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_80/ReadVariableOp(^batch_normalization_80/ReadVariableOp_17^batch_normalization_81/FusedBatchNormV3/ReadVariableOp9^batch_normalization_81/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_81/ReadVariableOp(^batch_normalization_81/ReadVariableOp_17^batch_normalization_82/FusedBatchNormV3/ReadVariableOp9^batch_normalization_82/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_82/ReadVariableOp(^batch_normalization_82/ReadVariableOp_17^batch_normalization_83/FusedBatchNormV3/ReadVariableOp9^batch_normalization_83/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_83/ReadVariableOp(^batch_normalization_83/ReadVariableOp_1!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp!^conv2d_67/BiasAdd/ReadVariableOp ^conv2d_67/Conv2D/ReadVariableOp!^conv2d_68/BiasAdd/ReadVariableOp ^conv2d_68/Conv2D/ReadVariableOp!^conv2d_69/BiasAdd/ReadVariableOp ^conv2d_69/Conv2D/ReadVariableOp!^conv2d_70/BiasAdd/ReadVariableOp ^conv2d_70/Conv2D/ReadVariableOp!^conv2d_71/BiasAdd/ReadVariableOp ^conv2d_71/Conv2D/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2p
6batch_normalization_77/FusedBatchNormV3/ReadVariableOp6batch_normalization_77/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_18batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_77/ReadVariableOp%batch_normalization_77/ReadVariableOp2R
'batch_normalization_77/ReadVariableOp_1'batch_normalization_77/ReadVariableOp_12p
6batch_normalization_78/FusedBatchNormV3/ReadVariableOp6batch_normalization_78/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_78/FusedBatchNormV3/ReadVariableOp_18batch_normalization_78/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_78/ReadVariableOp%batch_normalization_78/ReadVariableOp2R
'batch_normalization_78/ReadVariableOp_1'batch_normalization_78/ReadVariableOp_12p
6batch_normalization_79/FusedBatchNormV3/ReadVariableOp6batch_normalization_79/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_79/FusedBatchNormV3/ReadVariableOp_18batch_normalization_79/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_79/ReadVariableOp%batch_normalization_79/ReadVariableOp2R
'batch_normalization_79/ReadVariableOp_1'batch_normalization_79/ReadVariableOp_12p
6batch_normalization_80/FusedBatchNormV3/ReadVariableOp6batch_normalization_80/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_80/FusedBatchNormV3/ReadVariableOp_18batch_normalization_80/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_80/ReadVariableOp%batch_normalization_80/ReadVariableOp2R
'batch_normalization_80/ReadVariableOp_1'batch_normalization_80/ReadVariableOp_12p
6batch_normalization_81/FusedBatchNormV3/ReadVariableOp6batch_normalization_81/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_81/FusedBatchNormV3/ReadVariableOp_18batch_normalization_81/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_81/ReadVariableOp%batch_normalization_81/ReadVariableOp2R
'batch_normalization_81/ReadVariableOp_1'batch_normalization_81/ReadVariableOp_12p
6batch_normalization_82/FusedBatchNormV3/ReadVariableOp6batch_normalization_82/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_82/FusedBatchNormV3/ReadVariableOp_18batch_normalization_82/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_82/ReadVariableOp%batch_normalization_82/ReadVariableOp2R
'batch_normalization_82/ReadVariableOp_1'batch_normalization_82/ReadVariableOp_12p
6batch_normalization_83/FusedBatchNormV3/ReadVariableOp6batch_normalization_83/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_83/FusedBatchNormV3/ReadVariableOp_18batch_normalization_83/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_83/ReadVariableOp%batch_normalization_83/ReadVariableOp2R
'batch_normalization_83/ReadVariableOp_1'batch_normalization_83/ReadVariableOp_12D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2D
 conv2d_67/BiasAdd/ReadVariableOp conv2d_67/BiasAdd/ReadVariableOp2B
conv2d_67/Conv2D/ReadVariableOpconv2d_67/Conv2D/ReadVariableOp2D
 conv2d_68/BiasAdd/ReadVariableOp conv2d_68/BiasAdd/ReadVariableOp2B
conv2d_68/Conv2D/ReadVariableOpconv2d_68/Conv2D/ReadVariableOp2D
 conv2d_69/BiasAdd/ReadVariableOp conv2d_69/BiasAdd/ReadVariableOp2B
conv2d_69/Conv2D/ReadVariableOpconv2d_69/Conv2D/ReadVariableOp2D
 conv2d_70/BiasAdd/ReadVariableOp conv2d_70/BiasAdd/ReadVariableOp2B
conv2d_70/Conv2D/ReadVariableOpconv2d_70/Conv2D/ReadVariableOp2D
 conv2d_71/BiasAdd/ReadVariableOp conv2d_71/BiasAdd/ReadVariableOp2B
conv2d_71/Conv2D/ReadVariableOpconv2d_71/Conv2D/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_82_layer_call_fn_695901

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_6937172
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
�
h
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_692597

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
�j
�
I__inference_sequential_11_layer_call_and_return_conditional_losses_694011
conv2d_66_input
conv2d_66_693907
conv2d_66_693909!
batch_normalization_77_693912!
batch_normalization_77_693914!
batch_normalization_77_693916!
batch_normalization_77_693918
conv2d_67_693921
conv2d_67_693923!
batch_normalization_78_693926!
batch_normalization_78_693928!
batch_normalization_78_693930!
batch_normalization_78_693932
conv2d_68_693936
conv2d_68_693938!
batch_normalization_79_693941!
batch_normalization_79_693943!
batch_normalization_79_693945!
batch_normalization_79_693947
conv2d_69_693950
conv2d_69_693952!
batch_normalization_80_693955!
batch_normalization_80_693957!
batch_normalization_80_693959!
batch_normalization_80_693961
conv2d_70_693965
conv2d_70_693967!
batch_normalization_81_693970!
batch_normalization_81_693972!
batch_normalization_81_693974!
batch_normalization_81_693976
conv2d_71_693979
conv2d_71_693981!
batch_normalization_82_693984!
batch_normalization_82_693986!
batch_normalization_82_693988!
batch_normalization_82_693990!
batch_normalization_83_693994!
batch_normalization_83_693996!
batch_normalization_83_693998!
batch_normalization_83_694000
dense_11_694004
dense_11_694006
identity��.batch_normalization_77/StatefulPartitionedCall�.batch_normalization_78/StatefulPartitionedCall�.batch_normalization_79/StatefulPartitionedCall�.batch_normalization_80/StatefulPartitionedCall�.batch_normalization_81/StatefulPartitionedCall�.batch_normalization_82/StatefulPartitionedCall�.batch_normalization_83/StatefulPartitionedCall�!conv2d_66/StatefulPartitionedCall�!conv2d_67/StatefulPartitionedCall�!conv2d_68/StatefulPartitionedCall�!conv2d_69/StatefulPartitionedCall�!conv2d_70/StatefulPartitionedCall�!conv2d_71/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallconv2d_66_inputconv2d_66_693907conv2d_66_693909*
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
E__inference_conv2d_66_layer_call_and_return_conditional_losses_6931622#
!conv2d_66/StatefulPartitionedCall�
.batch_normalization_77/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_77_693912batch_normalization_77_693914batch_normalization_77_693916batch_normalization_77_693918*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_69321520
.batch_normalization_77/StatefulPartitionedCall�
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_77/StatefulPartitionedCall:output:0conv2d_67_693921conv2d_67_693923*
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
E__inference_conv2d_67_layer_call_and_return_conditional_losses_6932622#
!conv2d_67/StatefulPartitionedCall�
.batch_normalization_78/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_78_693926batch_normalization_78_693928batch_normalization_78_693930batch_normalization_78_693932*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_69331520
.batch_normalization_78/StatefulPartitionedCall�
 max_pooling2d_33/PartitionedCallPartitionedCall7batch_normalization_78/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_6925972"
 max_pooling2d_33/PartitionedCall�
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0conv2d_68_693936conv2d_68_693938*
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
E__inference_conv2d_68_layer_call_and_return_conditional_losses_6933632#
!conv2d_68/StatefulPartitionedCall�
.batch_normalization_79/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_79_693941batch_normalization_79_693943batch_normalization_79_693945batch_normalization_79_693947*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_69341620
.batch_normalization_79/StatefulPartitionedCall�
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_79/StatefulPartitionedCall:output:0conv2d_69_693950conv2d_69_693952*
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
E__inference_conv2d_69_layer_call_and_return_conditional_losses_6934632#
!conv2d_69/StatefulPartitionedCall�
.batch_normalization_80/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0batch_normalization_80_693955batch_normalization_80_693957batch_normalization_80_693959batch_normalization_80_693961*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_69351620
.batch_normalization_80/StatefulPartitionedCall�
 max_pooling2d_34/PartitionedCallPartitionedCall7batch_normalization_80/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_6928172"
 max_pooling2d_34/PartitionedCall�
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0conv2d_70_693965conv2d_70_693967*
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
E__inference_conv2d_70_layer_call_and_return_conditional_losses_6935642#
!conv2d_70/StatefulPartitionedCall�
.batch_normalization_81/StatefulPartitionedCallStatefulPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0batch_normalization_81_693970batch_normalization_81_693972batch_normalization_81_693974batch_normalization_81_693976*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_69361720
.batch_normalization_81/StatefulPartitionedCall�
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_81/StatefulPartitionedCall:output:0conv2d_71_693979conv2d_71_693981*
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
E__inference_conv2d_71_layer_call_and_return_conditional_losses_6936642#
!conv2d_71/StatefulPartitionedCall�
.batch_normalization_82/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0batch_normalization_82_693984batch_normalization_82_693986batch_normalization_82_693988batch_normalization_82_693990*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_69371720
.batch_normalization_82/StatefulPartitionedCall�
 max_pooling2d_35/PartitionedCallPartitionedCall7batch_normalization_82/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_6930372"
 max_pooling2d_35/PartitionedCall�
.batch_normalization_83/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_83_693994batch_normalization_83_693996batch_normalization_83_693998batch_normalization_83_694000*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_69379120
.batch_normalization_83/StatefulPartitionedCall�
flatten_11/PartitionedCallPartitionedCall7batch_normalization_83/StatefulPartitionedCall:output:0*
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
F__inference_flatten_11_layer_call_and_return_conditional_losses_6938332
flatten_11/PartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_11_694004dense_11_694006*
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
D__inference_dense_11_layer_call_and_return_conditional_losses_6938522"
 dense_11/StatefulPartitionedCall�
lambda_11/PartitionedCallPartitionedCall)dense_11/StatefulPartitionedCall:output:0*
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
E__inference_lambda_11_layer_call_and_return_conditional_losses_6938902
lambda_11/PartitionedCall�
IdentityIdentity"lambda_11/PartitionedCall:output:0/^batch_normalization_77/StatefulPartitionedCall/^batch_normalization_78/StatefulPartitionedCall/^batch_normalization_79/StatefulPartitionedCall/^batch_normalization_80/StatefulPartitionedCall/^batch_normalization_81/StatefulPartitionedCall/^batch_normalization_82/StatefulPartitionedCall/^batch_normalization_83/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_77/StatefulPartitionedCall.batch_normalization_77/StatefulPartitionedCall2`
.batch_normalization_78/StatefulPartitionedCall.batch_normalization_78/StatefulPartitionedCall2`
.batch_normalization_79/StatefulPartitionedCall.batch_normalization_79/StatefulPartitionedCall2`
.batch_normalization_80/StatefulPartitionedCall.batch_normalization_80/StatefulPartitionedCall2`
.batch_normalization_81/StatefulPartitionedCall.batch_normalization_81/StatefulPartitionedCall2`
.batch_normalization_82/StatefulPartitionedCall.batch_normalization_82/StatefulPartitionedCall2`
.batch_normalization_83/StatefulPartitionedCall.batch_normalization_83/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_66_input
�
�
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695367

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
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_692800

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
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_692989

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
E__inference_conv2d_71_layer_call_and_return_conditional_losses_695764

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
7__inference_batch_normalization_81_layer_call_fn_695676

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_6928852
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
7__inference_batch_normalization_77_layer_call_fn_695161

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_6924762
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
F
*__inference_lambda_11_layer_call_fn_696092

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
E__inference_lambda_11_layer_call_and_return_conditional_losses_6938902
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
�
�
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695283

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
�
�
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695579

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
�
b
F__inference_flatten_11_layer_call_and_return_conditional_losses_696035

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
conv2d_66_input@
!serving_default_conv2d_66_input:0���������  >
	lambda_111
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
	variables
trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"��
_tf_keras_sequential֣{"class_name": "Sequential", "name": "sequential_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_66_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_66", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_77", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_67", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_78", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_68", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_79", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_69", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_80", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_70", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_81", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_71", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_82", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_83", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Lambda", "config": {"name": "lambda_11", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPoe\nPGlweXRob24taW5wdXQtNS1kODA2ODAzY2Y1YTY+2gg8bGFtYmRhPhgAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_66_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_66", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_77", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_67", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_78", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_68", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_79", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_69", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_80", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_70", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_81", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_71", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_82", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_83", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Lambda", "config": {"name": "lambda_11", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPoe\nPGlweXRob24taW5wdXQtNS1kODA2ODAzY2Y1YTY+2gg8bGFtYmRhPhgAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}}}
�


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_66", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_66", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
�	
axis
	 gamma
!beta
"moving_mean
#moving_variance
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_77", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_77", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�	

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_67", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_67", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�	
.axis
	/gamma
0beta
1moving_mean
2moving_variance
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_78", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_78", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_33", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_68", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_68", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
�	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_79", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_79", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�	

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_69", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_69", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�	
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_80", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_80", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_34", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_70", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_70", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
�	
caxis
	dgamma
ebeta
fmoving_mean
gmoving_variance
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_81", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_81", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�	

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_71", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_71", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�	
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_82", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_82", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�
{regularization_losses
|	variables
}trainable_variables
~	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_35", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	
axis

�gamma
	�beta
�moving_mean
�moving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_83", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_83", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_11", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "lambda_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_11", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPoe\nPGlweXRob24taW5wdXQtNS1kODA2ODAzY2Y1YTY+2gg8bGFtYmRhPhgAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
 "
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
regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
	variables
�metrics
�layer_metrics
trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
*:( 2conv2d_66/kernel
: 2conv2d_66/bias
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
regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
	variables
�metrics
�layer_metrics
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_77/gamma
):' 2batch_normalization_77/beta
2:0  (2"batch_normalization_77/moving_mean
6:4  (2&batch_normalization_77/moving_variance
 "
trackable_list_wrapper
<
 0
!1
"2
#3"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
�
$regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
%	variables
�metrics
�layer_metrics
&trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_67/kernel
: 2conv2d_67/bias
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
*regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
+	variables
�metrics
�layer_metrics
,trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_78/gamma
):' 2batch_normalization_78/beta
2:0  (2"batch_normalization_78/moving_mean
6:4  (2&batch_normalization_78/moving_variance
 "
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
�
3regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
4	variables
�metrics
�layer_metrics
5trainable_variables
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
7regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
8	variables
�metrics
�layer_metrics
9trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_68/kernel
:@2conv2d_68/bias
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
=regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
>	variables
�metrics
�layer_metrics
?trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_79/gamma
):'@2batch_normalization_79/beta
2:0@ (2"batch_normalization_79/moving_mean
6:4@ (2&batch_normalization_79/moving_variance
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
�
Fregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
G	variables
�metrics
�layer_metrics
Htrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_69/kernel
:@2conv2d_69/bias
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
Lregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
M	variables
�metrics
�layer_metrics
Ntrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_80/gamma
):'@2batch_normalization_80/beta
2:0@ (2"batch_normalization_80/moving_mean
6:4@ (2&batch_normalization_80/moving_variance
 "
trackable_list_wrapper
<
Q0
R1
S2
T3"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
�
Uregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
V	variables
�metrics
�layer_metrics
Wtrainable_variables
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
Yregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
Z	variables
�metrics
�layer_metrics
[trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)@�2conv2d_70/kernel
:�2conv2d_70/bias
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
_regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
`	variables
�metrics
�layer_metrics
atrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)�2batch_normalization_81/gamma
*:(�2batch_normalization_81/beta
3:1� (2"batch_normalization_81/moving_mean
7:5� (2&batch_normalization_81/moving_variance
 "
trackable_list_wrapper
<
d0
e1
f2
g3"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
�
hregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
i	variables
�metrics
�layer_metrics
jtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*��2conv2d_71/kernel
:�2conv2d_71/bias
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
nregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
o	variables
�metrics
�layer_metrics
ptrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)�2batch_normalization_82/gamma
*:(�2batch_normalization_82/beta
3:1� (2"batch_normalization_82/moving_mean
7:5� (2&batch_normalization_82/moving_variance
 "
trackable_list_wrapper
<
s0
t1
u2
v3"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
�
wregularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
x	variables
�metrics
�layer_metrics
ytrainable_variables
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
{regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
|	variables
�metrics
�layer_metrics
}trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)�2batch_normalization_83/gamma
*:(�2batch_normalization_83/beta
3:1� (2"batch_normalization_83/moving_mean
7:5� (2&batch_normalization_83/moving_variance
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
�	variables
�metrics
�layer_metrics
�trainable_variables
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
�regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
�	variables
�metrics
�layer_metrics
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_11/kernel
:�2dense_11/bias
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
�regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
�	variables
�metrics
�layer_metrics
�trainable_variables
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
�regularization_losses
�layers
 �layer_regularization_losses
�non_trainable_variables
�	variables
�metrics
�layer_metrics
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
.
"0
#1"
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
.
10
21"
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
.
D0
E1"
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
.
S0
T1"
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
.
f0
g1"
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
.
u0
v1"
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
0
�0
�1"
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
 "
trackable_dict_wrapper
�2�
I__inference_sequential_11_layer_call_and_return_conditional_losses_694672
I__inference_sequential_11_layer_call_and_return_conditional_losses_694835
I__inference_sequential_11_layer_call_and_return_conditional_losses_694011
I__inference_sequential_11_layer_call_and_return_conditional_losses_693904�
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
!__inference__wrapped_model_692383�
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
conv2d_66_input���������  
�2�
.__inference_sequential_11_layer_call_fn_694924
.__inference_sequential_11_layer_call_fn_695013
.__inference_sequential_11_layer_call_fn_694208
.__inference_sequential_11_layer_call_fn_694404�
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
E__inference_conv2d_66_layer_call_and_return_conditional_losses_695024�
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
*__inference_conv2d_66_layer_call_fn_695033�
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
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695135
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695053
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695117
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695071�
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
7__inference_batch_normalization_77_layer_call_fn_695161
7__inference_batch_normalization_77_layer_call_fn_695084
7__inference_batch_normalization_77_layer_call_fn_695097
7__inference_batch_normalization_77_layer_call_fn_695148�
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
E__inference_conv2d_67_layer_call_and_return_conditional_losses_695172�
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
*__inference_conv2d_67_layer_call_fn_695181�
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
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695219
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695283
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695265
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695201�
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
7__inference_batch_normalization_78_layer_call_fn_695296
7__inference_batch_normalization_78_layer_call_fn_695232
7__inference_batch_normalization_78_layer_call_fn_695309
7__inference_batch_normalization_78_layer_call_fn_695245�
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
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_692597�
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
1__inference_max_pooling2d_33_layer_call_fn_692603�
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
E__inference_conv2d_68_layer_call_and_return_conditional_losses_695320�
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
*__inference_conv2d_68_layer_call_fn_695329�
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
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695431
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695367
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695413
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695349�
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
7__inference_batch_normalization_79_layer_call_fn_695380
7__inference_batch_normalization_79_layer_call_fn_695393
7__inference_batch_normalization_79_layer_call_fn_695457
7__inference_batch_normalization_79_layer_call_fn_695444�
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
E__inference_conv2d_69_layer_call_and_return_conditional_losses_695468�
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
*__inference_conv2d_69_layer_call_fn_695477�
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
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695515
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695579
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695561
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695497�
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
7__inference_batch_normalization_80_layer_call_fn_695592
7__inference_batch_normalization_80_layer_call_fn_695605
7__inference_batch_normalization_80_layer_call_fn_695541
7__inference_batch_normalization_80_layer_call_fn_695528�
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
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_692817�
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
1__inference_max_pooling2d_34_layer_call_fn_692823�
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
E__inference_conv2d_70_layer_call_and_return_conditional_losses_695616�
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
*__inference_conv2d_70_layer_call_fn_695625�
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
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695645
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695663
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695727
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695709�
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
7__inference_batch_normalization_81_layer_call_fn_695689
7__inference_batch_normalization_81_layer_call_fn_695676
7__inference_batch_normalization_81_layer_call_fn_695753
7__inference_batch_normalization_81_layer_call_fn_695740�
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
E__inference_conv2d_71_layer_call_and_return_conditional_losses_695764�
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
*__inference_conv2d_71_layer_call_fn_695773�
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
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695811
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695793
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695875
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695857�
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
7__inference_batch_normalization_82_layer_call_fn_695901
7__inference_batch_normalization_82_layer_call_fn_695888
7__inference_batch_normalization_82_layer_call_fn_695824
7__inference_batch_normalization_82_layer_call_fn_695837�
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
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_693037�
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
1__inference_max_pooling2d_35_layer_call_fn_693043�
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
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_695921
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_695985
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_696003
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_695939�
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
7__inference_batch_normalization_83_layer_call_fn_695965
7__inference_batch_normalization_83_layer_call_fn_695952
7__inference_batch_normalization_83_layer_call_fn_696029
7__inference_batch_normalization_83_layer_call_fn_696016�
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
F__inference_flatten_11_layer_call_and_return_conditional_losses_696035�
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
+__inference_flatten_11_layer_call_fn_696040�
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
D__inference_dense_11_layer_call_and_return_conditional_losses_696051�
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
)__inference_dense_11_layer_call_fn_696060�
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
E__inference_lambda_11_layer_call_and_return_conditional_losses_696082
E__inference_lambda_11_layer_call_and_return_conditional_losses_696071�
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
*__inference_lambda_11_layer_call_fn_696087
*__inference_lambda_11_layer_call_fn_696092�
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
$__inference_signature_wrapper_694495conv2d_66_input"�
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
!__inference__wrapped_model_692383�0 !"#()/012;<BCDEJKQRST]^defglmstuv������@�=
6�3
1�.
conv2d_66_input���������  
� "6�3
1
	lambda_11$�!
	lambda_11�����������
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695053r !"#;�8
1�.
(�%
inputs���������   
p
� "-�*
#� 
0���������   
� �
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695071r !"#;�8
1�.
(�%
inputs���������   
p 
� "-�*
#� 
0���������   
� �
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695117� !"#M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_77_layer_call_and_return_conditional_losses_695135� !"#M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
7__inference_batch_normalization_77_layer_call_fn_695084e !"#;�8
1�.
(�%
inputs���������   
p
� " ����������   �
7__inference_batch_normalization_77_layer_call_fn_695097e !"#;�8
1�.
(�%
inputs���������   
p 
� " ����������   �
7__inference_batch_normalization_77_layer_call_fn_695148� !"#M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
7__inference_batch_normalization_77_layer_call_fn_695161� !"#M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695201�/012M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695219�/012M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695265r/012;�8
1�.
(�%
inputs���������   
p
� "-�*
#� 
0���������   
� �
R__inference_batch_normalization_78_layer_call_and_return_conditional_losses_695283r/012;�8
1�.
(�%
inputs���������   
p 
� "-�*
#� 
0���������   
� �
7__inference_batch_normalization_78_layer_call_fn_695232�/012M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
7__inference_batch_normalization_78_layer_call_fn_695245�/012M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
7__inference_batch_normalization_78_layer_call_fn_695296e/012;�8
1�.
(�%
inputs���������   
p
� " ����������   �
7__inference_batch_normalization_78_layer_call_fn_695309e/012;�8
1�.
(�%
inputs���������   
p 
� " ����������   �
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695349�BCDEM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695367�BCDEM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695413rBCDE;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_79_layer_call_and_return_conditional_losses_695431rBCDE;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
7__inference_batch_normalization_79_layer_call_fn_695380�BCDEM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_79_layer_call_fn_695393�BCDEM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_79_layer_call_fn_695444eBCDE;�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_79_layer_call_fn_695457eBCDE;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695497�QRSTM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695515�QRSTM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695561rQRST;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_80_layer_call_and_return_conditional_losses_695579rQRST;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
7__inference_batch_normalization_80_layer_call_fn_695528�QRSTM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_80_layer_call_fn_695541�QRSTM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_80_layer_call_fn_695592eQRST;�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_80_layer_call_fn_695605eQRST;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695645�defgN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695663�defgN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695709tdefg<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
R__inference_batch_normalization_81_layer_call_and_return_conditional_losses_695727tdefg<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
7__inference_batch_normalization_81_layer_call_fn_695676�defgN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
7__inference_batch_normalization_81_layer_call_fn_695689�defgN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
7__inference_batch_normalization_81_layer_call_fn_695740gdefg<�9
2�/
)�&
inputs����������
p
� "!������������
7__inference_batch_normalization_81_layer_call_fn_695753gdefg<�9
2�/
)�&
inputs����������
p 
� "!������������
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695793�stuvN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695811�stuvN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695857tstuv<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
R__inference_batch_normalization_82_layer_call_and_return_conditional_losses_695875tstuv<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
7__inference_batch_normalization_82_layer_call_fn_695824�stuvN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
7__inference_batch_normalization_82_layer_call_fn_695837�stuvN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
7__inference_batch_normalization_82_layer_call_fn_695888gstuv<�9
2�/
)�&
inputs����������
p
� "!������������
7__inference_batch_normalization_82_layer_call_fn_695901gstuv<�9
2�/
)�&
inputs����������
p 
� "!������������
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_695921�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_695939�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_695985x����<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
R__inference_batch_normalization_83_layer_call_and_return_conditional_losses_696003x����<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
7__inference_batch_normalization_83_layer_call_fn_695952�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
7__inference_batch_normalization_83_layer_call_fn_695965�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
7__inference_batch_normalization_83_layer_call_fn_696016k����<�9
2�/
)�&
inputs����������
p
� "!������������
7__inference_batch_normalization_83_layer_call_fn_696029k����<�9
2�/
)�&
inputs����������
p 
� "!������������
E__inference_conv2d_66_layer_call_and_return_conditional_losses_695024l7�4
-�*
(�%
inputs���������  
� "-�*
#� 
0���������   
� �
*__inference_conv2d_66_layer_call_fn_695033_7�4
-�*
(�%
inputs���������  
� " ����������   �
E__inference_conv2d_67_layer_call_and_return_conditional_losses_695172l()7�4
-�*
(�%
inputs���������   
� "-�*
#� 
0���������   
� �
*__inference_conv2d_67_layer_call_fn_695181_()7�4
-�*
(�%
inputs���������   
� " ����������   �
E__inference_conv2d_68_layer_call_and_return_conditional_losses_695320l;<7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
*__inference_conv2d_68_layer_call_fn_695329_;<7�4
-�*
(�%
inputs��������� 
� " ����������@�
E__inference_conv2d_69_layer_call_and_return_conditional_losses_695468lJK7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
*__inference_conv2d_69_layer_call_fn_695477_JK7�4
-�*
(�%
inputs���������@
� " ����������@�
E__inference_conv2d_70_layer_call_and_return_conditional_losses_695616m]^7�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
*__inference_conv2d_70_layer_call_fn_695625`]^7�4
-�*
(�%
inputs���������@
� "!������������
E__inference_conv2d_71_layer_call_and_return_conditional_losses_695764nlm8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
*__inference_conv2d_71_layer_call_fn_695773alm8�5
.�+
)�&
inputs����������
� "!������������
D__inference_dense_11_layer_call_and_return_conditional_losses_696051`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
)__inference_dense_11_layer_call_fn_696060S��0�-
&�#
!�
inputs����������
� "������������
F__inference_flatten_11_layer_call_and_return_conditional_losses_696035b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
+__inference_flatten_11_layer_call_fn_696040U8�5
.�+
)�&
inputs����������
� "������������
E__inference_lambda_11_layer_call_and_return_conditional_losses_696071b8�5
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
E__inference_lambda_11_layer_call_and_return_conditional_losses_696082b8�5
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
*__inference_lambda_11_layer_call_fn_696087U8�5
.�+
!�
inputs����������

 
p
� "������������
*__inference_lambda_11_layer_call_fn_696092U8�5
.�+
!�
inputs����������

 
p 
� "������������
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_692597�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_33_layer_call_fn_692603�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_692817�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_34_layer_call_fn_692823�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_693037�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_35_layer_call_fn_693043�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_11_layer_call_and_return_conditional_losses_693904�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_66_input���������  
p

 
� "&�#
�
0����������
� �
I__inference_sequential_11_layer_call_and_return_conditional_losses_694011�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_66_input���������  
p 

 
� "&�#
�
0����������
� �
I__inference_sequential_11_layer_call_and_return_conditional_losses_694672�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
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
I__inference_sequential_11_layer_call_and_return_conditional_losses_694835�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
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
.__inference_sequential_11_layer_call_fn_694208�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_66_input���������  
p

 
� "������������
.__inference_sequential_11_layer_call_fn_694404�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_66_input���������  
p 

 
� "������������
.__inference_sequential_11_layer_call_fn_694924�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
5�2
(�%
inputs���������  
p

 
� "������������
.__inference_sequential_11_layer_call_fn_695013�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
5�2
(�%
inputs���������  
p 

 
� "������������
$__inference_signature_wrapper_694495�0 !"#()/012;<BCDEJKQRST]^defglmstuv������S�P
� 
I�F
D
conv2d_66_input1�.
conv2d_66_input���������  "6�3
1
	lambda_11$�!
	lambda_11����������