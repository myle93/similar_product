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
conv2d_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_72/kernel
}
$conv2d_72/kernel/Read/ReadVariableOpReadVariableOpconv2d_72/kernel*&
_output_shapes
: *
dtype0
t
conv2d_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_72/bias
m
"conv2d_72/bias/Read/ReadVariableOpReadVariableOpconv2d_72/bias*
_output_shapes
: *
dtype0
�
batch_normalization_84/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_84/gamma
�
0batch_normalization_84/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_84/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_84/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_84/beta
�
/batch_normalization_84/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_84/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_84/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_84/moving_mean
�
6batch_normalization_84/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_84/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_84/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_84/moving_variance
�
:batch_normalization_84/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_84/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_73/kernel
}
$conv2d_73/kernel/Read/ReadVariableOpReadVariableOpconv2d_73/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_73/bias
m
"conv2d_73/bias/Read/ReadVariableOpReadVariableOpconv2d_73/bias*
_output_shapes
: *
dtype0
�
batch_normalization_85/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_85/gamma
�
0batch_normalization_85/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_85/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_85/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_85/beta
�
/batch_normalization_85/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_85/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_85/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_85/moving_mean
�
6batch_normalization_85/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_85/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_85/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_85/moving_variance
�
:batch_normalization_85/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_85/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_74/kernel
}
$conv2d_74/kernel/Read/ReadVariableOpReadVariableOpconv2d_74/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_74/bias
m
"conv2d_74/bias/Read/ReadVariableOpReadVariableOpconv2d_74/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_86/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_86/gamma
�
0batch_normalization_86/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_86/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_86/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_86/beta
�
/batch_normalization_86/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_86/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_86/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_86/moving_mean
�
6batch_normalization_86/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_86/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_86/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_86/moving_variance
�
:batch_normalization_86/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_86/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_75/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_75/kernel
}
$conv2d_75/kernel/Read/ReadVariableOpReadVariableOpconv2d_75/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_75/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_75/bias
m
"conv2d_75/bias/Read/ReadVariableOpReadVariableOpconv2d_75/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_87/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_87/gamma
�
0batch_normalization_87/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_87/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_87/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_87/beta
�
/batch_normalization_87/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_87/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_87/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_87/moving_mean
�
6batch_normalization_87/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_87/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_87/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_87/moving_variance
�
:batch_normalization_87/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_87/moving_variance*
_output_shapes
:@*
dtype0
�
conv2d_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameconv2d_76/kernel
~
$conv2d_76/kernel/Read/ReadVariableOpReadVariableOpconv2d_76/kernel*'
_output_shapes
:@�*
dtype0
u
conv2d_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_76/bias
n
"conv2d_76/bias/Read/ReadVariableOpReadVariableOpconv2d_76/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_88/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_88/gamma
�
0batch_normalization_88/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_88/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_88/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_88/beta
�
/batch_normalization_88/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_88/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_88/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_88/moving_mean
�
6batch_normalization_88/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_88/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_88/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_88/moving_variance
�
:batch_normalization_88/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_88/moving_variance*
_output_shapes	
:�*
dtype0
�
conv2d_77/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_77/kernel

$conv2d_77/kernel/Read/ReadVariableOpReadVariableOpconv2d_77/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_77/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_77/bias
n
"conv2d_77/bias/Read/ReadVariableOpReadVariableOpconv2d_77/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_89/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_89/gamma
�
0batch_normalization_89/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_89/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_89/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_89/beta
�
/batch_normalization_89/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_89/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_89/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_89/moving_mean
�
6batch_normalization_89/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_89/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_89/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_89/moving_variance
�
:batch_normalization_89/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_89/moving_variance*
_output_shapes	
:�*
dtype0
�
batch_normalization_90/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_90/gamma
�
0batch_normalization_90/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_90/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_90/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_90/beta
�
/batch_normalization_90/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_90/beta*
_output_shapes	
:�*
dtype0
�
"batch_normalization_90/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_90/moving_mean
�
6batch_normalization_90/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_90/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_90/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_90/moving_variance
�
:batch_normalization_90/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_90/moving_variance*
_output_shapes	
:�*
dtype0
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
��*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
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
VARIABLE_VALUEconv2d_72/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_72/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_84/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_84/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_84/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_84/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_73/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_73/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_85/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_85/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_85/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_85/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_74/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_74/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_86/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_86/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_86/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_86/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_75/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_75/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_87/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_87/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_87/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_87/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_76/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_76/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_88/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_88/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_88/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_88/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_77/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_77/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_89/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_89/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_89/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_89/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEbatch_normalization_90/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_90/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_90/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_90/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_12/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_12/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_conv2d_72_inputPlaceholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_72_inputconv2d_72/kernelconv2d_72/biasbatch_normalization_84/gammabatch_normalization_84/beta"batch_normalization_84/moving_mean&batch_normalization_84/moving_varianceconv2d_73/kernelconv2d_73/biasbatch_normalization_85/gammabatch_normalization_85/beta"batch_normalization_85/moving_mean&batch_normalization_85/moving_varianceconv2d_74/kernelconv2d_74/biasbatch_normalization_86/gammabatch_normalization_86/beta"batch_normalization_86/moving_mean&batch_normalization_86/moving_varianceconv2d_75/kernelconv2d_75/biasbatch_normalization_87/gammabatch_normalization_87/beta"batch_normalization_87/moving_mean&batch_normalization_87/moving_varianceconv2d_76/kernelconv2d_76/biasbatch_normalization_88/gammabatch_normalization_88/beta"batch_normalization_88/moving_mean&batch_normalization_88/moving_varianceconv2d_77/kernelconv2d_77/biasbatch_normalization_89/gammabatch_normalization_89/beta"batch_normalization_89/moving_mean&batch_normalization_89/moving_variancebatch_normalization_90/gammabatch_normalization_90/beta"batch_normalization_90/moving_mean&batch_normalization_90/moving_variancedense_12/kerneldense_12/bias*6
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
$__inference_signature_wrapper_796090
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_72/kernel/Read/ReadVariableOp"conv2d_72/bias/Read/ReadVariableOp0batch_normalization_84/gamma/Read/ReadVariableOp/batch_normalization_84/beta/Read/ReadVariableOp6batch_normalization_84/moving_mean/Read/ReadVariableOp:batch_normalization_84/moving_variance/Read/ReadVariableOp$conv2d_73/kernel/Read/ReadVariableOp"conv2d_73/bias/Read/ReadVariableOp0batch_normalization_85/gamma/Read/ReadVariableOp/batch_normalization_85/beta/Read/ReadVariableOp6batch_normalization_85/moving_mean/Read/ReadVariableOp:batch_normalization_85/moving_variance/Read/ReadVariableOp$conv2d_74/kernel/Read/ReadVariableOp"conv2d_74/bias/Read/ReadVariableOp0batch_normalization_86/gamma/Read/ReadVariableOp/batch_normalization_86/beta/Read/ReadVariableOp6batch_normalization_86/moving_mean/Read/ReadVariableOp:batch_normalization_86/moving_variance/Read/ReadVariableOp$conv2d_75/kernel/Read/ReadVariableOp"conv2d_75/bias/Read/ReadVariableOp0batch_normalization_87/gamma/Read/ReadVariableOp/batch_normalization_87/beta/Read/ReadVariableOp6batch_normalization_87/moving_mean/Read/ReadVariableOp:batch_normalization_87/moving_variance/Read/ReadVariableOp$conv2d_76/kernel/Read/ReadVariableOp"conv2d_76/bias/Read/ReadVariableOp0batch_normalization_88/gamma/Read/ReadVariableOp/batch_normalization_88/beta/Read/ReadVariableOp6batch_normalization_88/moving_mean/Read/ReadVariableOp:batch_normalization_88/moving_variance/Read/ReadVariableOp$conv2d_77/kernel/Read/ReadVariableOp"conv2d_77/bias/Read/ReadVariableOp0batch_normalization_89/gamma/Read/ReadVariableOp/batch_normalization_89/beta/Read/ReadVariableOp6batch_normalization_89/moving_mean/Read/ReadVariableOp:batch_normalization_89/moving_variance/Read/ReadVariableOp0batch_normalization_90/gamma/Read/ReadVariableOp/batch_normalization_90/beta/Read/ReadVariableOp6batch_normalization_90/moving_mean/Read/ReadVariableOp:batch_normalization_90/moving_variance/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOpConst*7
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
__inference__traced_save_797836
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_72/kernelconv2d_72/biasbatch_normalization_84/gammabatch_normalization_84/beta"batch_normalization_84/moving_mean&batch_normalization_84/moving_varianceconv2d_73/kernelconv2d_73/biasbatch_normalization_85/gammabatch_normalization_85/beta"batch_normalization_85/moving_mean&batch_normalization_85/moving_varianceconv2d_74/kernelconv2d_74/biasbatch_normalization_86/gammabatch_normalization_86/beta"batch_normalization_86/moving_mean&batch_normalization_86/moving_varianceconv2d_75/kernelconv2d_75/biasbatch_normalization_87/gammabatch_normalization_87/beta"batch_normalization_87/moving_mean&batch_normalization_87/moving_varianceconv2d_76/kernelconv2d_76/biasbatch_normalization_88/gammabatch_normalization_88/beta"batch_normalization_88/moving_mean&batch_normalization_88/moving_varianceconv2d_77/kernelconv2d_77/biasbatch_normalization_89/gammabatch_normalization_89/beta"batch_normalization_89/moving_mean&batch_normalization_89/moving_variancebatch_normalization_90/gammabatch_normalization_90/beta"batch_normalization_90/moving_mean&batch_normalization_90/moving_variancedense_12/kerneldense_12/bias*6
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
"__inference__traced_restore_797972��
ҳ
�
"__inference__traced_restore_797972
file_prefix%
!assignvariableop_conv2d_72_kernel%
!assignvariableop_1_conv2d_72_bias3
/assignvariableop_2_batch_normalization_84_gamma2
.assignvariableop_3_batch_normalization_84_beta9
5assignvariableop_4_batch_normalization_84_moving_mean=
9assignvariableop_5_batch_normalization_84_moving_variance'
#assignvariableop_6_conv2d_73_kernel%
!assignvariableop_7_conv2d_73_bias3
/assignvariableop_8_batch_normalization_85_gamma2
.assignvariableop_9_batch_normalization_85_beta:
6assignvariableop_10_batch_normalization_85_moving_mean>
:assignvariableop_11_batch_normalization_85_moving_variance(
$assignvariableop_12_conv2d_74_kernel&
"assignvariableop_13_conv2d_74_bias4
0assignvariableop_14_batch_normalization_86_gamma3
/assignvariableop_15_batch_normalization_86_beta:
6assignvariableop_16_batch_normalization_86_moving_mean>
:assignvariableop_17_batch_normalization_86_moving_variance(
$assignvariableop_18_conv2d_75_kernel&
"assignvariableop_19_conv2d_75_bias4
0assignvariableop_20_batch_normalization_87_gamma3
/assignvariableop_21_batch_normalization_87_beta:
6assignvariableop_22_batch_normalization_87_moving_mean>
:assignvariableop_23_batch_normalization_87_moving_variance(
$assignvariableop_24_conv2d_76_kernel&
"assignvariableop_25_conv2d_76_bias4
0assignvariableop_26_batch_normalization_88_gamma3
/assignvariableop_27_batch_normalization_88_beta:
6assignvariableop_28_batch_normalization_88_moving_mean>
:assignvariableop_29_batch_normalization_88_moving_variance(
$assignvariableop_30_conv2d_77_kernel&
"assignvariableop_31_conv2d_77_bias4
0assignvariableop_32_batch_normalization_89_gamma3
/assignvariableop_33_batch_normalization_89_beta:
6assignvariableop_34_batch_normalization_89_moving_mean>
:assignvariableop_35_batch_normalization_89_moving_variance4
0assignvariableop_36_batch_normalization_90_gamma3
/assignvariableop_37_batch_normalization_90_beta:
6assignvariableop_38_batch_normalization_90_moving_mean>
:assignvariableop_39_batch_normalization_90_moving_variance'
#assignvariableop_40_dense_12_kernel%
!assignvariableop_41_dense_12_bias
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_72_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_72_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_84_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_84_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_84_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_84_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_73_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_73_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_85_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_85_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_85_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_85_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_74_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_74_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_86_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_86_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_86_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_86_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_75_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_75_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_87_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_87_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_87_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_87_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_76_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_76_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_88_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_88_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_88_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_88_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_77_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_77_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_89_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_89_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_89_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_89_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_90_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp/assignvariableop_37_batch_normalization_90_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp6assignvariableop_38_batch_normalization_90_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp:assignvariableop_39_batch_normalization_90_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp#assignvariableop_40_dense_12_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp!assignvariableop_41_dense_12_biasIdentity_41:output:0"/device:CPU:0*
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
�

a
E__inference_lambda_12_layer_call_and_return_conditional_losses_797666

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
�
�
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797388

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
.__inference_sequential_12_layer_call_fn_796519

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
I__inference_sequential_12_layer_call_and_return_conditional_losses_7957162
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
�
�
7__inference_batch_normalization_89_layer_call_fn_797496

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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_7953122
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

*__inference_conv2d_75_layer_call_fn_797072

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
E__inference_conv2d_75_layer_call_and_return_conditional_losses_7950582
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
�
�
7__inference_batch_normalization_87_layer_call_fn_797123

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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7943642
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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_795294

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
E__inference_conv2d_72_layer_call_and_return_conditional_losses_796619

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
�
�
7__inference_batch_normalization_84_layer_call_fn_796756

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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7940712
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
E__inference_conv2d_72_layer_call_and_return_conditional_losses_794757

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
�

�
E__inference_conv2d_74_layer_call_and_return_conditional_losses_794958

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
�
M
1__inference_max_pooling2d_37_layer_call_fn_794418

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
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_7944122
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
�
F
*__inference_lambda_12_layer_call_fn_797682

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
E__inference_lambda_12_layer_call_and_return_conditional_losses_7954742
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
�
�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_794731

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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796878

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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_794615

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
M
1__inference_max_pooling2d_36_layer_call_fn_794198

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
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_7941922
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
E__inference_conv2d_75_layer_call_and_return_conditional_losses_795058

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
�

�
E__inference_conv2d_75_layer_call_and_return_conditional_losses_797063

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
7__inference_batch_normalization_86_layer_call_fn_797039

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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7949932
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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_794364

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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_794071

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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_794700

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
7__inference_batch_normalization_88_layer_call_fn_797284

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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7945112
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
E__inference_conv2d_73_layer_call_and_return_conditional_losses_794857

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
�
�
7__inference_batch_normalization_89_layer_call_fn_797432

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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_7946152
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
E__inference_conv2d_77_layer_call_and_return_conditional_losses_797359

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
�
�
$__inference_signature_wrapper_796090
conv2d_72_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_72_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_7939782
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
_user_specified_nameconv2d_72_input
�
�
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_795011

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

*__inference_conv2d_76_layer_call_fn_797220

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
E__inference_conv2d_76_layer_call_and_return_conditional_losses_7951592
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
�
b
F__inference_flatten_12_layer_call_and_return_conditional_losses_797630

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
�
�
7__inference_batch_normalization_85_layer_call_fn_796891

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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7948922
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
7__inference_batch_normalization_89_layer_call_fn_797419

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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_7945842
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

a
E__inference_lambda_12_layer_call_and_return_conditional_losses_797677

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
7__inference_batch_normalization_87_layer_call_fn_797200

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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7951112
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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796796

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
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_794192

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
I__inference_sequential_12_layer_call_and_return_conditional_losses_795912

inputs
conv2d_72_795808
conv2d_72_795810!
batch_normalization_84_795813!
batch_normalization_84_795815!
batch_normalization_84_795817!
batch_normalization_84_795819
conv2d_73_795822
conv2d_73_795824!
batch_normalization_85_795827!
batch_normalization_85_795829!
batch_normalization_85_795831!
batch_normalization_85_795833
conv2d_74_795837
conv2d_74_795839!
batch_normalization_86_795842!
batch_normalization_86_795844!
batch_normalization_86_795846!
batch_normalization_86_795848
conv2d_75_795851
conv2d_75_795853!
batch_normalization_87_795856!
batch_normalization_87_795858!
batch_normalization_87_795860!
batch_normalization_87_795862
conv2d_76_795866
conv2d_76_795868!
batch_normalization_88_795871!
batch_normalization_88_795873!
batch_normalization_88_795875!
batch_normalization_88_795877
conv2d_77_795880
conv2d_77_795882!
batch_normalization_89_795885!
batch_normalization_89_795887!
batch_normalization_89_795889!
batch_normalization_89_795891!
batch_normalization_90_795895!
batch_normalization_90_795897!
batch_normalization_90_795899!
batch_normalization_90_795901
dense_12_795905
dense_12_795907
identity��.batch_normalization_84/StatefulPartitionedCall�.batch_normalization_85/StatefulPartitionedCall�.batch_normalization_86/StatefulPartitionedCall�.batch_normalization_87/StatefulPartitionedCall�.batch_normalization_88/StatefulPartitionedCall�.batch_normalization_89/StatefulPartitionedCall�.batch_normalization_90/StatefulPartitionedCall�!conv2d_72/StatefulPartitionedCall�!conv2d_73/StatefulPartitionedCall�!conv2d_74/StatefulPartitionedCall�!conv2d_75/StatefulPartitionedCall�!conv2d_76/StatefulPartitionedCall�!conv2d_77/StatefulPartitionedCall� dense_12/StatefulPartitionedCall�
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_72_795808conv2d_72_795810*
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
E__inference_conv2d_72_layer_call_and_return_conditional_losses_7947572#
!conv2d_72/StatefulPartitionedCall�
.batch_normalization_84/StatefulPartitionedCallStatefulPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0batch_normalization_84_795813batch_normalization_84_795815batch_normalization_84_795817batch_normalization_84_795819*
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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_79481020
.batch_normalization_84/StatefulPartitionedCall�
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_84/StatefulPartitionedCall:output:0conv2d_73_795822conv2d_73_795824*
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
E__inference_conv2d_73_layer_call_and_return_conditional_losses_7948572#
!conv2d_73/StatefulPartitionedCall�
.batch_normalization_85/StatefulPartitionedCallStatefulPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0batch_normalization_85_795827batch_normalization_85_795829batch_normalization_85_795831batch_normalization_85_795833*
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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_79491020
.batch_normalization_85/StatefulPartitionedCall�
 max_pooling2d_36/PartitionedCallPartitionedCall7batch_normalization_85/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_7941922"
 max_pooling2d_36/PartitionedCall�
!conv2d_74/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_74_795837conv2d_74_795839*
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
E__inference_conv2d_74_layer_call_and_return_conditional_losses_7949582#
!conv2d_74/StatefulPartitionedCall�
.batch_normalization_86/StatefulPartitionedCallStatefulPartitionedCall*conv2d_74/StatefulPartitionedCall:output:0batch_normalization_86_795842batch_normalization_86_795844batch_normalization_86_795846batch_normalization_86_795848*
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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_79501120
.batch_normalization_86/StatefulPartitionedCall�
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_86/StatefulPartitionedCall:output:0conv2d_75_795851conv2d_75_795853*
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
E__inference_conv2d_75_layer_call_and_return_conditional_losses_7950582#
!conv2d_75/StatefulPartitionedCall�
.batch_normalization_87/StatefulPartitionedCallStatefulPartitionedCall*conv2d_75/StatefulPartitionedCall:output:0batch_normalization_87_795856batch_normalization_87_795858batch_normalization_87_795860batch_normalization_87_795862*
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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_79511120
.batch_normalization_87/StatefulPartitionedCall�
 max_pooling2d_37/PartitionedCallPartitionedCall7batch_normalization_87/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_7944122"
 max_pooling2d_37/PartitionedCall�
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_76_795866conv2d_76_795868*
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
E__inference_conv2d_76_layer_call_and_return_conditional_losses_7951592#
!conv2d_76/StatefulPartitionedCall�
.batch_normalization_88/StatefulPartitionedCallStatefulPartitionedCall*conv2d_76/StatefulPartitionedCall:output:0batch_normalization_88_795871batch_normalization_88_795873batch_normalization_88_795875batch_normalization_88_795877*
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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_79521220
.batch_normalization_88/StatefulPartitionedCall�
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_88/StatefulPartitionedCall:output:0conv2d_77_795880conv2d_77_795882*
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
E__inference_conv2d_77_layer_call_and_return_conditional_losses_7952592#
!conv2d_77/StatefulPartitionedCall�
.batch_normalization_89/StatefulPartitionedCallStatefulPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0batch_normalization_89_795885batch_normalization_89_795887batch_normalization_89_795889batch_normalization_89_795891*
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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_79531220
.batch_normalization_89/StatefulPartitionedCall�
 max_pooling2d_38/PartitionedCallPartitionedCall7batch_normalization_89/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_7946322"
 max_pooling2d_38/PartitionedCall�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0batch_normalization_90_795895batch_normalization_90_795897batch_normalization_90_795899batch_normalization_90_795901*
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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_79538620
.batch_normalization_90/StatefulPartitionedCall�
flatten_12/PartitionedCallPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0*
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
F__inference_flatten_12_layer_call_and_return_conditional_losses_7954282
flatten_12/PartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0dense_12_795905dense_12_795907*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_7954472"
 dense_12/StatefulPartitionedCall�
lambda_12/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
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
E__inference_lambda_12_layer_call_and_return_conditional_losses_7954852
lambda_12/PartitionedCall�
IdentityIdentity"lambda_12/PartitionedCall:output:0/^batch_normalization_84/StatefulPartitionedCall/^batch_normalization_85/StatefulPartitionedCall/^batch_normalization_86/StatefulPartitionedCall/^batch_normalization_87/StatefulPartitionedCall/^batch_normalization_88/StatefulPartitionedCall/^batch_normalization_89/StatefulPartitionedCall/^batch_normalization_90/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall"^conv2d_74/StatefulPartitionedCall"^conv2d_75/StatefulPartitionedCall"^conv2d_76/StatefulPartitionedCall"^conv2d_77/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_84/StatefulPartitionedCall.batch_normalization_84/StatefulPartitionedCall2`
.batch_normalization_85/StatefulPartitionedCall.batch_normalization_85/StatefulPartitionedCall2`
.batch_normalization_86/StatefulPartitionedCall.batch_normalization_86/StatefulPartitionedCall2`
.batch_normalization_87/StatefulPartitionedCall.batch_normalization_87/StatefulPartitionedCall2`
.batch_normalization_88/StatefulPartitionedCall.batch_normalization_88/StatefulPartitionedCall2`
.batch_normalization_89/StatefulPartitionedCall.batch_normalization_89/StatefulPartitionedCall2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2F
!conv2d_74/StatefulPartitionedCall!conv2d_74/StatefulPartitionedCall2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�

a
E__inference_lambda_12_layer_call_and_return_conditional_losses_795485

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
�
�
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_795093

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
��
�!
I__inference_sequential_12_layer_call_and_return_conditional_losses_796430

inputs,
(conv2d_72_conv2d_readvariableop_resource-
)conv2d_72_biasadd_readvariableop_resource2
.batch_normalization_84_readvariableop_resource4
0batch_normalization_84_readvariableop_1_resourceC
?batch_normalization_84_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_84_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_73_conv2d_readvariableop_resource-
)conv2d_73_biasadd_readvariableop_resource2
.batch_normalization_85_readvariableop_resource4
0batch_normalization_85_readvariableop_1_resourceC
?batch_normalization_85_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_85_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_74_conv2d_readvariableop_resource-
)conv2d_74_biasadd_readvariableop_resource2
.batch_normalization_86_readvariableop_resource4
0batch_normalization_86_readvariableop_1_resourceC
?batch_normalization_86_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_86_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_75_conv2d_readvariableop_resource-
)conv2d_75_biasadd_readvariableop_resource2
.batch_normalization_87_readvariableop_resource4
0batch_normalization_87_readvariableop_1_resourceC
?batch_normalization_87_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_87_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_76_conv2d_readvariableop_resource-
)conv2d_76_biasadd_readvariableop_resource2
.batch_normalization_88_readvariableop_resource4
0batch_normalization_88_readvariableop_1_resourceC
?batch_normalization_88_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_88_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_77_conv2d_readvariableop_resource-
)conv2d_77_biasadd_readvariableop_resource2
.batch_normalization_89_readvariableop_resource4
0batch_normalization_89_readvariableop_1_resourceC
?batch_normalization_89_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_89_fusedbatchnormv3_readvariableop_1_resource2
.batch_normalization_90_readvariableop_resource4
0batch_normalization_90_readvariableop_1_resourceC
?batch_normalization_90_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_90_fusedbatchnormv3_readvariableop_1_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity��6batch_normalization_84/FusedBatchNormV3/ReadVariableOp�8batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_84/ReadVariableOp�'batch_normalization_84/ReadVariableOp_1�6batch_normalization_85/FusedBatchNormV3/ReadVariableOp�8batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_85/ReadVariableOp�'batch_normalization_85/ReadVariableOp_1�6batch_normalization_86/FusedBatchNormV3/ReadVariableOp�8batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_86/ReadVariableOp�'batch_normalization_86/ReadVariableOp_1�6batch_normalization_87/FusedBatchNormV3/ReadVariableOp�8batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_87/ReadVariableOp�'batch_normalization_87/ReadVariableOp_1�6batch_normalization_88/FusedBatchNormV3/ReadVariableOp�8batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_88/ReadVariableOp�'batch_normalization_88/ReadVariableOp_1�6batch_normalization_89/FusedBatchNormV3/ReadVariableOp�8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_89/ReadVariableOp�'batch_normalization_89/ReadVariableOp_1�6batch_normalization_90/FusedBatchNormV3/ReadVariableOp�8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_90/ReadVariableOp�'batch_normalization_90/ReadVariableOp_1� conv2d_72/BiasAdd/ReadVariableOp�conv2d_72/Conv2D/ReadVariableOp� conv2d_73/BiasAdd/ReadVariableOp�conv2d_73/Conv2D/ReadVariableOp� conv2d_74/BiasAdd/ReadVariableOp�conv2d_74/Conv2D/ReadVariableOp� conv2d_75/BiasAdd/ReadVariableOp�conv2d_75/Conv2D/ReadVariableOp� conv2d_76/BiasAdd/ReadVariableOp�conv2d_76/Conv2D/ReadVariableOp� conv2d_77/BiasAdd/ReadVariableOp�conv2d_77/Conv2D/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�
conv2d_72/Conv2D/ReadVariableOpReadVariableOp(conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_72/Conv2D/ReadVariableOp�
conv2d_72/Conv2DConv2Dinputs'conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_72/Conv2D�
 conv2d_72/BiasAdd/ReadVariableOpReadVariableOp)conv2d_72_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_72/BiasAdd/ReadVariableOp�
conv2d_72/BiasAddBiasAddconv2d_72/Conv2D:output:0(conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_72/BiasAdd~
conv2d_72/ReluReluconv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_72/Relu�
%batch_normalization_84/ReadVariableOpReadVariableOp.batch_normalization_84_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_84/ReadVariableOp�
'batch_normalization_84/ReadVariableOp_1ReadVariableOp0batch_normalization_84_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_84/ReadVariableOp_1�
6batch_normalization_84/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_84_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_84/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_84_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_84/FusedBatchNormV3FusedBatchNormV3conv2d_72/Relu:activations:0-batch_normalization_84/ReadVariableOp:value:0/batch_normalization_84/ReadVariableOp_1:value:0>batch_normalization_84/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_84/FusedBatchNormV3�
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_73/Conv2D/ReadVariableOp�
conv2d_73/Conv2DConv2D+batch_normalization_84/FusedBatchNormV3:y:0'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_73/Conv2D�
 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_73/BiasAdd/ReadVariableOp�
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_73/BiasAdd~
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_73/Relu�
%batch_normalization_85/ReadVariableOpReadVariableOp.batch_normalization_85_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_85/ReadVariableOp�
'batch_normalization_85/ReadVariableOp_1ReadVariableOp0batch_normalization_85_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_85/ReadVariableOp_1�
6batch_normalization_85/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_85_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_85/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_85_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_85/FusedBatchNormV3FusedBatchNormV3conv2d_73/Relu:activations:0-batch_normalization_85/ReadVariableOp:value:0/batch_normalization_85/ReadVariableOp_1:value:0>batch_normalization_85/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2)
'batch_normalization_85/FusedBatchNormV3�
max_pooling2d_36/MaxPoolMaxPool+batch_normalization_85/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPool�
conv2d_74/Conv2D/ReadVariableOpReadVariableOp(conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_74/Conv2D/ReadVariableOp�
conv2d_74/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_74/Conv2D�
 conv2d_74/BiasAdd/ReadVariableOpReadVariableOp)conv2d_74_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_74/BiasAdd/ReadVariableOp�
conv2d_74/BiasAddBiasAddconv2d_74/Conv2D:output:0(conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_74/BiasAdd~
conv2d_74/ReluReluconv2d_74/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_74/Relu�
%batch_normalization_86/ReadVariableOpReadVariableOp.batch_normalization_86_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_86/ReadVariableOp�
'batch_normalization_86/ReadVariableOp_1ReadVariableOp0batch_normalization_86_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_86/ReadVariableOp_1�
6batch_normalization_86/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_86_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_86/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_86_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_86/FusedBatchNormV3FusedBatchNormV3conv2d_74/Relu:activations:0-batch_normalization_86/ReadVariableOp:value:0/batch_normalization_86/ReadVariableOp_1:value:0>batch_normalization_86/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_86/FusedBatchNormV3�
conv2d_75/Conv2D/ReadVariableOpReadVariableOp(conv2d_75_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_75/Conv2D/ReadVariableOp�
conv2d_75/Conv2DConv2D+batch_normalization_86/FusedBatchNormV3:y:0'conv2d_75/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_75/Conv2D�
 conv2d_75/BiasAdd/ReadVariableOpReadVariableOp)conv2d_75_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_75/BiasAdd/ReadVariableOp�
conv2d_75/BiasAddBiasAddconv2d_75/Conv2D:output:0(conv2d_75/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_75/BiasAdd~
conv2d_75/ReluReluconv2d_75/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_75/Relu�
%batch_normalization_87/ReadVariableOpReadVariableOp.batch_normalization_87_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_87/ReadVariableOp�
'batch_normalization_87/ReadVariableOp_1ReadVariableOp0batch_normalization_87_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_87/ReadVariableOp_1�
6batch_normalization_87/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_87_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_87/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_87_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_87/FusedBatchNormV3FusedBatchNormV3conv2d_75/Relu:activations:0-batch_normalization_87/ReadVariableOp:value:0/batch_normalization_87/ReadVariableOp_1:value:0>batch_normalization_87/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_87/FusedBatchNormV3�
max_pooling2d_37/MaxPoolMaxPool+batch_normalization_87/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPool�
conv2d_76/Conv2D/ReadVariableOpReadVariableOp(conv2d_76_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_76/Conv2D/ReadVariableOp�
conv2d_76/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_76/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_76/Conv2D�
 conv2d_76/BiasAdd/ReadVariableOpReadVariableOp)conv2d_76_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_76/BiasAdd/ReadVariableOp�
conv2d_76/BiasAddBiasAddconv2d_76/Conv2D:output:0(conv2d_76/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_76/BiasAdd
conv2d_76/ReluReluconv2d_76/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_76/Relu�
%batch_normalization_88/ReadVariableOpReadVariableOp.batch_normalization_88_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_88/ReadVariableOp�
'batch_normalization_88/ReadVariableOp_1ReadVariableOp0batch_normalization_88_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_88/ReadVariableOp_1�
6batch_normalization_88/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_88_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_88/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_88_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_88/FusedBatchNormV3FusedBatchNormV3conv2d_76/Relu:activations:0-batch_normalization_88/ReadVariableOp:value:0/batch_normalization_88/ReadVariableOp_1:value:0>batch_normalization_88/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_88/FusedBatchNormV3�
conv2d_77/Conv2D/ReadVariableOpReadVariableOp(conv2d_77_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_77/Conv2D/ReadVariableOp�
conv2d_77/Conv2DConv2D+batch_normalization_88/FusedBatchNormV3:y:0'conv2d_77/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_77/Conv2D�
 conv2d_77/BiasAdd/ReadVariableOpReadVariableOp)conv2d_77_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_77/BiasAdd/ReadVariableOp�
conv2d_77/BiasAddBiasAddconv2d_77/Conv2D:output:0(conv2d_77/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_77/BiasAdd
conv2d_77/ReluReluconv2d_77/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_77/Relu�
%batch_normalization_89/ReadVariableOpReadVariableOp.batch_normalization_89_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_89/ReadVariableOp�
'batch_normalization_89/ReadVariableOp_1ReadVariableOp0batch_normalization_89_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_89/ReadVariableOp_1�
6batch_normalization_89/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_89_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_89/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_89_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_89/FusedBatchNormV3FusedBatchNormV3conv2d_77/Relu:activations:0-batch_normalization_89/ReadVariableOp:value:0/batch_normalization_89/ReadVariableOp_1:value:0>batch_normalization_89/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_89/FusedBatchNormV3�
max_pooling2d_38/MaxPoolMaxPool+batch_normalization_89/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool�
%batch_normalization_90/ReadVariableOpReadVariableOp.batch_normalization_90_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_90/ReadVariableOp�
'batch_normalization_90/ReadVariableOp_1ReadVariableOp0batch_normalization_90_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_90/ReadVariableOp_1�
6batch_normalization_90/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_90_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_90/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_90_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_90/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_38/MaxPool:output:0-batch_normalization_90/ReadVariableOp:value:0/batch_normalization_90/ReadVariableOp_1:value:0>batch_normalization_90/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2)
'batch_normalization_90/FusedBatchNormV3u
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_12/Const�
flatten_12/ReshapeReshape+batch_normalization_90/FusedBatchNormV3:y:0flatten_12/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_12/Reshape�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulflatten_12/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAdd}
dense_12/SigmoidSigmoiddense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_12/Sigmoid�
lambda_12/l2_normalize/SquareSquaredense_12/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
lambda_12/l2_normalize/Square�
,lambda_12/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,lambda_12/l2_normalize/Sum/reduction_indices�
lambda_12/l2_normalize/SumSum!lambda_12/l2_normalize/Square:y:05lambda_12/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
lambda_12/l2_normalize/Sum�
 lambda_12/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2"
 lambda_12/l2_normalize/Maximum/y�
lambda_12/l2_normalize/MaximumMaximum#lambda_12/l2_normalize/Sum:output:0)lambda_12/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2 
lambda_12/l2_normalize/Maximum�
lambda_12/l2_normalize/RsqrtRsqrt"lambda_12/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
lambda_12/l2_normalize/Rsqrt�
lambda_12/l2_normalizeMuldense_12/Sigmoid:y:0 lambda_12/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
lambda_12/l2_normalize�
IdentityIdentitylambda_12/l2_normalize:z:07^batch_normalization_84/FusedBatchNormV3/ReadVariableOp9^batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_84/ReadVariableOp(^batch_normalization_84/ReadVariableOp_17^batch_normalization_85/FusedBatchNormV3/ReadVariableOp9^batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_85/ReadVariableOp(^batch_normalization_85/ReadVariableOp_17^batch_normalization_86/FusedBatchNormV3/ReadVariableOp9^batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_86/ReadVariableOp(^batch_normalization_86/ReadVariableOp_17^batch_normalization_87/FusedBatchNormV3/ReadVariableOp9^batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_87/ReadVariableOp(^batch_normalization_87/ReadVariableOp_17^batch_normalization_88/FusedBatchNormV3/ReadVariableOp9^batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_88/ReadVariableOp(^batch_normalization_88/ReadVariableOp_17^batch_normalization_89/FusedBatchNormV3/ReadVariableOp9^batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_89/ReadVariableOp(^batch_normalization_89/ReadVariableOp_17^batch_normalization_90/FusedBatchNormV3/ReadVariableOp9^batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_90/ReadVariableOp(^batch_normalization_90/ReadVariableOp_1!^conv2d_72/BiasAdd/ReadVariableOp ^conv2d_72/Conv2D/ReadVariableOp!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp!^conv2d_74/BiasAdd/ReadVariableOp ^conv2d_74/Conv2D/ReadVariableOp!^conv2d_75/BiasAdd/ReadVariableOp ^conv2d_75/Conv2D/ReadVariableOp!^conv2d_76/BiasAdd/ReadVariableOp ^conv2d_76/Conv2D/ReadVariableOp!^conv2d_77/BiasAdd/ReadVariableOp ^conv2d_77/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2p
6batch_normalization_84/FusedBatchNormV3/ReadVariableOp6batch_normalization_84/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_84/FusedBatchNormV3/ReadVariableOp_18batch_normalization_84/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_84/ReadVariableOp%batch_normalization_84/ReadVariableOp2R
'batch_normalization_84/ReadVariableOp_1'batch_normalization_84/ReadVariableOp_12p
6batch_normalization_85/FusedBatchNormV3/ReadVariableOp6batch_normalization_85/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_85/FusedBatchNormV3/ReadVariableOp_18batch_normalization_85/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_85/ReadVariableOp%batch_normalization_85/ReadVariableOp2R
'batch_normalization_85/ReadVariableOp_1'batch_normalization_85/ReadVariableOp_12p
6batch_normalization_86/FusedBatchNormV3/ReadVariableOp6batch_normalization_86/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_86/FusedBatchNormV3/ReadVariableOp_18batch_normalization_86/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_86/ReadVariableOp%batch_normalization_86/ReadVariableOp2R
'batch_normalization_86/ReadVariableOp_1'batch_normalization_86/ReadVariableOp_12p
6batch_normalization_87/FusedBatchNormV3/ReadVariableOp6batch_normalization_87/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_87/FusedBatchNormV3/ReadVariableOp_18batch_normalization_87/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_87/ReadVariableOp%batch_normalization_87/ReadVariableOp2R
'batch_normalization_87/ReadVariableOp_1'batch_normalization_87/ReadVariableOp_12p
6batch_normalization_88/FusedBatchNormV3/ReadVariableOp6batch_normalization_88/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_88/FusedBatchNormV3/ReadVariableOp_18batch_normalization_88/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_88/ReadVariableOp%batch_normalization_88/ReadVariableOp2R
'batch_normalization_88/ReadVariableOp_1'batch_normalization_88/ReadVariableOp_12p
6batch_normalization_89/FusedBatchNormV3/ReadVariableOp6batch_normalization_89/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_18batch_normalization_89/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_89/ReadVariableOp%batch_normalization_89/ReadVariableOp2R
'batch_normalization_89/ReadVariableOp_1'batch_normalization_89/ReadVariableOp_12p
6batch_normalization_90/FusedBatchNormV3/ReadVariableOp6batch_normalization_90/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_18batch_normalization_90/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_90/ReadVariableOp%batch_normalization_90/ReadVariableOp2R
'batch_normalization_90/ReadVariableOp_1'batch_normalization_90/ReadVariableOp_12D
 conv2d_72/BiasAdd/ReadVariableOp conv2d_72/BiasAdd/ReadVariableOp2B
conv2d_72/Conv2D/ReadVariableOpconv2d_72/Conv2D/ReadVariableOp2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp2D
 conv2d_74/BiasAdd/ReadVariableOp conv2d_74/BiasAdd/ReadVariableOp2B
conv2d_74/Conv2D/ReadVariableOpconv2d_74/Conv2D/ReadVariableOp2D
 conv2d_75/BiasAdd/ReadVariableOp conv2d_75/BiasAdd/ReadVariableOp2B
conv2d_75/Conv2D/ReadVariableOpconv2d_75/Conv2D/ReadVariableOp2D
 conv2d_76/BiasAdd/ReadVariableOp conv2d_76/BiasAdd/ReadVariableOp2B
conv2d_76/Conv2D/ReadVariableOpconv2d_76/Conv2D/ReadVariableOp2D
 conv2d_77/BiasAdd/ReadVariableOp conv2d_77/BiasAdd/ReadVariableOp2B
conv2d_77/Conv2D/ReadVariableOpconv2d_77/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797598

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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797110

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
7__inference_batch_normalization_89_layer_call_fn_797483

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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_7952942
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
7__inference_batch_normalization_90_layer_call_fn_797611

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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_7953682
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
�j
�
I__inference_sequential_12_layer_call_and_return_conditional_losses_795499
conv2d_72_input
conv2d_72_794768
conv2d_72_794770!
batch_normalization_84_794837!
batch_normalization_84_794839!
batch_normalization_84_794841!
batch_normalization_84_794843
conv2d_73_794868
conv2d_73_794870!
batch_normalization_85_794937!
batch_normalization_85_794939!
batch_normalization_85_794941!
batch_normalization_85_794943
conv2d_74_794969
conv2d_74_794971!
batch_normalization_86_795038!
batch_normalization_86_795040!
batch_normalization_86_795042!
batch_normalization_86_795044
conv2d_75_795069
conv2d_75_795071!
batch_normalization_87_795138!
batch_normalization_87_795140!
batch_normalization_87_795142!
batch_normalization_87_795144
conv2d_76_795170
conv2d_76_795172!
batch_normalization_88_795239!
batch_normalization_88_795241!
batch_normalization_88_795243!
batch_normalization_88_795245
conv2d_77_795270
conv2d_77_795272!
batch_normalization_89_795339!
batch_normalization_89_795341!
batch_normalization_89_795343!
batch_normalization_89_795345!
batch_normalization_90_795413!
batch_normalization_90_795415!
batch_normalization_90_795417!
batch_normalization_90_795419
dense_12_795458
dense_12_795460
identity��.batch_normalization_84/StatefulPartitionedCall�.batch_normalization_85/StatefulPartitionedCall�.batch_normalization_86/StatefulPartitionedCall�.batch_normalization_87/StatefulPartitionedCall�.batch_normalization_88/StatefulPartitionedCall�.batch_normalization_89/StatefulPartitionedCall�.batch_normalization_90/StatefulPartitionedCall�!conv2d_72/StatefulPartitionedCall�!conv2d_73/StatefulPartitionedCall�!conv2d_74/StatefulPartitionedCall�!conv2d_75/StatefulPartitionedCall�!conv2d_76/StatefulPartitionedCall�!conv2d_77/StatefulPartitionedCall� dense_12/StatefulPartitionedCall�
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCallconv2d_72_inputconv2d_72_794768conv2d_72_794770*
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
E__inference_conv2d_72_layer_call_and_return_conditional_losses_7947572#
!conv2d_72/StatefulPartitionedCall�
.batch_normalization_84/StatefulPartitionedCallStatefulPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0batch_normalization_84_794837batch_normalization_84_794839batch_normalization_84_794841batch_normalization_84_794843*
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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_79479220
.batch_normalization_84/StatefulPartitionedCall�
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_84/StatefulPartitionedCall:output:0conv2d_73_794868conv2d_73_794870*
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
E__inference_conv2d_73_layer_call_and_return_conditional_losses_7948572#
!conv2d_73/StatefulPartitionedCall�
.batch_normalization_85/StatefulPartitionedCallStatefulPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0batch_normalization_85_794937batch_normalization_85_794939batch_normalization_85_794941batch_normalization_85_794943*
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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_79489220
.batch_normalization_85/StatefulPartitionedCall�
 max_pooling2d_36/PartitionedCallPartitionedCall7batch_normalization_85/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_7941922"
 max_pooling2d_36/PartitionedCall�
!conv2d_74/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_74_794969conv2d_74_794971*
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
E__inference_conv2d_74_layer_call_and_return_conditional_losses_7949582#
!conv2d_74/StatefulPartitionedCall�
.batch_normalization_86/StatefulPartitionedCallStatefulPartitionedCall*conv2d_74/StatefulPartitionedCall:output:0batch_normalization_86_795038batch_normalization_86_795040batch_normalization_86_795042batch_normalization_86_795044*
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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_79499320
.batch_normalization_86/StatefulPartitionedCall�
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_86/StatefulPartitionedCall:output:0conv2d_75_795069conv2d_75_795071*
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
E__inference_conv2d_75_layer_call_and_return_conditional_losses_7950582#
!conv2d_75/StatefulPartitionedCall�
.batch_normalization_87/StatefulPartitionedCallStatefulPartitionedCall*conv2d_75/StatefulPartitionedCall:output:0batch_normalization_87_795138batch_normalization_87_795140batch_normalization_87_795142batch_normalization_87_795144*
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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_79509320
.batch_normalization_87/StatefulPartitionedCall�
 max_pooling2d_37/PartitionedCallPartitionedCall7batch_normalization_87/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_7944122"
 max_pooling2d_37/PartitionedCall�
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_76_795170conv2d_76_795172*
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
E__inference_conv2d_76_layer_call_and_return_conditional_losses_7951592#
!conv2d_76/StatefulPartitionedCall�
.batch_normalization_88/StatefulPartitionedCallStatefulPartitionedCall*conv2d_76/StatefulPartitionedCall:output:0batch_normalization_88_795239batch_normalization_88_795241batch_normalization_88_795243batch_normalization_88_795245*
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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_79519420
.batch_normalization_88/StatefulPartitionedCall�
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_88/StatefulPartitionedCall:output:0conv2d_77_795270conv2d_77_795272*
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
E__inference_conv2d_77_layer_call_and_return_conditional_losses_7952592#
!conv2d_77/StatefulPartitionedCall�
.batch_normalization_89/StatefulPartitionedCallStatefulPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0batch_normalization_89_795339batch_normalization_89_795341batch_normalization_89_795343batch_normalization_89_795345*
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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_79529420
.batch_normalization_89/StatefulPartitionedCall�
 max_pooling2d_38/PartitionedCallPartitionedCall7batch_normalization_89/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_7946322"
 max_pooling2d_38/PartitionedCall�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0batch_normalization_90_795413batch_normalization_90_795415batch_normalization_90_795417batch_normalization_90_795419*
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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_79536820
.batch_normalization_90/StatefulPartitionedCall�
flatten_12/PartitionedCallPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0*
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
F__inference_flatten_12_layer_call_and_return_conditional_losses_7954282
flatten_12/PartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0dense_12_795458dense_12_795460*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_7954472"
 dense_12/StatefulPartitionedCall�
lambda_12/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
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
E__inference_lambda_12_layer_call_and_return_conditional_losses_7954742
lambda_12/PartitionedCall�
IdentityIdentity"lambda_12/PartitionedCall:output:0/^batch_normalization_84/StatefulPartitionedCall/^batch_normalization_85/StatefulPartitionedCall/^batch_normalization_86/StatefulPartitionedCall/^batch_normalization_87/StatefulPartitionedCall/^batch_normalization_88/StatefulPartitionedCall/^batch_normalization_89/StatefulPartitionedCall/^batch_normalization_90/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall"^conv2d_74/StatefulPartitionedCall"^conv2d_75/StatefulPartitionedCall"^conv2d_76/StatefulPartitionedCall"^conv2d_77/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_84/StatefulPartitionedCall.batch_normalization_84/StatefulPartitionedCall2`
.batch_normalization_85/StatefulPartitionedCall.batch_normalization_85/StatefulPartitionedCall2`
.batch_normalization_86/StatefulPartitionedCall.batch_normalization_86/StatefulPartitionedCall2`
.batch_normalization_87/StatefulPartitionedCall.batch_normalization_87/StatefulPartitionedCall2`
.batch_normalization_88/StatefulPartitionedCall.batch_normalization_88/StatefulPartitionedCall2`
.batch_normalization_89/StatefulPartitionedCall.batch_normalization_89/StatefulPartitionedCall2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2F
!conv2d_74/StatefulPartitionedCall!conv2d_74/StatefulPartitionedCall2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_72_input
�
�
7__inference_batch_normalization_88_layer_call_fn_797335

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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7951942
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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796712

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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_794260

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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797258

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

*__inference_conv2d_77_layer_call_fn_797368

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
E__inference_conv2d_77_layer_call_and_return_conditional_losses_7952592
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
�
�
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_794480

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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796814

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
7__inference_batch_normalization_85_layer_call_fn_796827

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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7941442
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
E__inference_conv2d_74_layer_call_and_return_conditional_losses_796915

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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797156

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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_794291

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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_794395

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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_794584

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
�

*__inference_conv2d_74_layer_call_fn_796924

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
E__inference_conv2d_74_layer_call_and_return_conditional_losses_7949582
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_795606
conv2d_72_input
conv2d_72_795502
conv2d_72_795504!
batch_normalization_84_795507!
batch_normalization_84_795509!
batch_normalization_84_795511!
batch_normalization_84_795513
conv2d_73_795516
conv2d_73_795518!
batch_normalization_85_795521!
batch_normalization_85_795523!
batch_normalization_85_795525!
batch_normalization_85_795527
conv2d_74_795531
conv2d_74_795533!
batch_normalization_86_795536!
batch_normalization_86_795538!
batch_normalization_86_795540!
batch_normalization_86_795542
conv2d_75_795545
conv2d_75_795547!
batch_normalization_87_795550!
batch_normalization_87_795552!
batch_normalization_87_795554!
batch_normalization_87_795556
conv2d_76_795560
conv2d_76_795562!
batch_normalization_88_795565!
batch_normalization_88_795567!
batch_normalization_88_795569!
batch_normalization_88_795571
conv2d_77_795574
conv2d_77_795576!
batch_normalization_89_795579!
batch_normalization_89_795581!
batch_normalization_89_795583!
batch_normalization_89_795585!
batch_normalization_90_795589!
batch_normalization_90_795591!
batch_normalization_90_795593!
batch_normalization_90_795595
dense_12_795599
dense_12_795601
identity��.batch_normalization_84/StatefulPartitionedCall�.batch_normalization_85/StatefulPartitionedCall�.batch_normalization_86/StatefulPartitionedCall�.batch_normalization_87/StatefulPartitionedCall�.batch_normalization_88/StatefulPartitionedCall�.batch_normalization_89/StatefulPartitionedCall�.batch_normalization_90/StatefulPartitionedCall�!conv2d_72/StatefulPartitionedCall�!conv2d_73/StatefulPartitionedCall�!conv2d_74/StatefulPartitionedCall�!conv2d_75/StatefulPartitionedCall�!conv2d_76/StatefulPartitionedCall�!conv2d_77/StatefulPartitionedCall� dense_12/StatefulPartitionedCall�
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCallconv2d_72_inputconv2d_72_795502conv2d_72_795504*
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
E__inference_conv2d_72_layer_call_and_return_conditional_losses_7947572#
!conv2d_72/StatefulPartitionedCall�
.batch_normalization_84/StatefulPartitionedCallStatefulPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0batch_normalization_84_795507batch_normalization_84_795509batch_normalization_84_795511batch_normalization_84_795513*
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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_79481020
.batch_normalization_84/StatefulPartitionedCall�
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_84/StatefulPartitionedCall:output:0conv2d_73_795516conv2d_73_795518*
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
E__inference_conv2d_73_layer_call_and_return_conditional_losses_7948572#
!conv2d_73/StatefulPartitionedCall�
.batch_normalization_85/StatefulPartitionedCallStatefulPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0batch_normalization_85_795521batch_normalization_85_795523batch_normalization_85_795525batch_normalization_85_795527*
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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_79491020
.batch_normalization_85/StatefulPartitionedCall�
 max_pooling2d_36/PartitionedCallPartitionedCall7batch_normalization_85/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_7941922"
 max_pooling2d_36/PartitionedCall�
!conv2d_74/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_74_795531conv2d_74_795533*
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
E__inference_conv2d_74_layer_call_and_return_conditional_losses_7949582#
!conv2d_74/StatefulPartitionedCall�
.batch_normalization_86/StatefulPartitionedCallStatefulPartitionedCall*conv2d_74/StatefulPartitionedCall:output:0batch_normalization_86_795536batch_normalization_86_795538batch_normalization_86_795540batch_normalization_86_795542*
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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_79501120
.batch_normalization_86/StatefulPartitionedCall�
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_86/StatefulPartitionedCall:output:0conv2d_75_795545conv2d_75_795547*
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
E__inference_conv2d_75_layer_call_and_return_conditional_losses_7950582#
!conv2d_75/StatefulPartitionedCall�
.batch_normalization_87/StatefulPartitionedCallStatefulPartitionedCall*conv2d_75/StatefulPartitionedCall:output:0batch_normalization_87_795550batch_normalization_87_795552batch_normalization_87_795554batch_normalization_87_795556*
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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_79511120
.batch_normalization_87/StatefulPartitionedCall�
 max_pooling2d_37/PartitionedCallPartitionedCall7batch_normalization_87/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_7944122"
 max_pooling2d_37/PartitionedCall�
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_76_795560conv2d_76_795562*
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
E__inference_conv2d_76_layer_call_and_return_conditional_losses_7951592#
!conv2d_76/StatefulPartitionedCall�
.batch_normalization_88/StatefulPartitionedCallStatefulPartitionedCall*conv2d_76/StatefulPartitionedCall:output:0batch_normalization_88_795565batch_normalization_88_795567batch_normalization_88_795569batch_normalization_88_795571*
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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_79521220
.batch_normalization_88/StatefulPartitionedCall�
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_88/StatefulPartitionedCall:output:0conv2d_77_795574conv2d_77_795576*
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
E__inference_conv2d_77_layer_call_and_return_conditional_losses_7952592#
!conv2d_77/StatefulPartitionedCall�
.batch_normalization_89/StatefulPartitionedCallStatefulPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0batch_normalization_89_795579batch_normalization_89_795581batch_normalization_89_795583batch_normalization_89_795585*
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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_79531220
.batch_normalization_89/StatefulPartitionedCall�
 max_pooling2d_38/PartitionedCallPartitionedCall7batch_normalization_89/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_7946322"
 max_pooling2d_38/PartitionedCall�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0batch_normalization_90_795589batch_normalization_90_795591batch_normalization_90_795593batch_normalization_90_795595*
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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_79538620
.batch_normalization_90/StatefulPartitionedCall�
flatten_12/PartitionedCallPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0*
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
F__inference_flatten_12_layer_call_and_return_conditional_losses_7954282
flatten_12/PartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0dense_12_795599dense_12_795601*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_7954472"
 dense_12/StatefulPartitionedCall�
lambda_12/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
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
E__inference_lambda_12_layer_call_and_return_conditional_losses_7954852
lambda_12/PartitionedCall�
IdentityIdentity"lambda_12/PartitionedCall:output:0/^batch_normalization_84/StatefulPartitionedCall/^batch_normalization_85/StatefulPartitionedCall/^batch_normalization_86/StatefulPartitionedCall/^batch_normalization_87/StatefulPartitionedCall/^batch_normalization_88/StatefulPartitionedCall/^batch_normalization_89/StatefulPartitionedCall/^batch_normalization_90/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall"^conv2d_74/StatefulPartitionedCall"^conv2d_75/StatefulPartitionedCall"^conv2d_76/StatefulPartitionedCall"^conv2d_77/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_84/StatefulPartitionedCall.batch_normalization_84/StatefulPartitionedCall2`
.batch_normalization_85/StatefulPartitionedCall.batch_normalization_85/StatefulPartitionedCall2`
.batch_normalization_86/StatefulPartitionedCall.batch_normalization_86/StatefulPartitionedCall2`
.batch_normalization_87/StatefulPartitionedCall.batch_normalization_87/StatefulPartitionedCall2`
.batch_normalization_88/StatefulPartitionedCall.batch_normalization_88/StatefulPartitionedCall2`
.batch_normalization_89/StatefulPartitionedCall.batch_normalization_89/StatefulPartitionedCall2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2F
!conv2d_74/StatefulPartitionedCall!conv2d_74/StatefulPartitionedCall2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_72_input
�
�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797534

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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797470

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
7__inference_batch_normalization_90_layer_call_fn_797560

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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_7947312
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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796666

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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_796962

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
�
�
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_794910

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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_794175

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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_795386

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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_795212

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
�
h
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_794632

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
�
�
7__inference_batch_normalization_84_layer_call_fn_796692

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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7948102
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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797092

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
�
�
.__inference_sequential_12_layer_call_fn_795803
conv2d_72_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_72_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_7957162
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
_user_specified_nameconv2d_72_input
�
�
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797174

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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_794792

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
�
�
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_797026

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
7__inference_batch_normalization_84_layer_call_fn_796679

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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7947922
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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797240

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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797452

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
7__inference_batch_normalization_86_layer_call_fn_797052

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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7950112
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
7__inference_batch_normalization_90_layer_call_fn_797547

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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_7947002
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
�
�
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_794993

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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_795312

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
�
.__inference_sequential_12_layer_call_fn_796608

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
I__inference_sequential_12_layer_call_and_return_conditional_losses_7959122
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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796860

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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797322

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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_794810

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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_794144

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
�
.__inference_sequential_12_layer_call_fn_795999
conv2d_72_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_72_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_7959122
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
_user_specified_nameconv2d_72_input
�
�
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_797008

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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_795194

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
7__inference_batch_normalization_86_layer_call_fn_796988

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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7942912
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
�
�
7__inference_batch_normalization_85_layer_call_fn_796840

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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7941752
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
E__inference_conv2d_77_layer_call_and_return_conditional_losses_795259

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
7__inference_batch_normalization_84_layer_call_fn_796743

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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_7940402
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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_794040

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
7__inference_batch_normalization_87_layer_call_fn_797187

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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7950932
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
7__inference_batch_normalization_88_layer_call_fn_797271

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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7944802
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
�
�
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_796944

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
E__inference_conv2d_76_layer_call_and_return_conditional_losses_797211

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
�

*__inference_conv2d_73_layer_call_fn_796776

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
E__inference_conv2d_73_layer_call_and_return_conditional_losses_7948572
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
�
�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797516

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
�
F
*__inference_lambda_12_layer_call_fn_797687

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
E__inference_lambda_12_layer_call_and_return_conditional_losses_7954852
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
�
h
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_794412

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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796730

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
�	
�
D__inference_dense_12_layer_call_and_return_conditional_losses_797646

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
�Z
�
__inference__traced_save_797836
file_prefix/
+savev2_conv2d_72_kernel_read_readvariableop-
)savev2_conv2d_72_bias_read_readvariableop;
7savev2_batch_normalization_84_gamma_read_readvariableop:
6savev2_batch_normalization_84_beta_read_readvariableopA
=savev2_batch_normalization_84_moving_mean_read_readvariableopE
Asavev2_batch_normalization_84_moving_variance_read_readvariableop/
+savev2_conv2d_73_kernel_read_readvariableop-
)savev2_conv2d_73_bias_read_readvariableop;
7savev2_batch_normalization_85_gamma_read_readvariableop:
6savev2_batch_normalization_85_beta_read_readvariableopA
=savev2_batch_normalization_85_moving_mean_read_readvariableopE
Asavev2_batch_normalization_85_moving_variance_read_readvariableop/
+savev2_conv2d_74_kernel_read_readvariableop-
)savev2_conv2d_74_bias_read_readvariableop;
7savev2_batch_normalization_86_gamma_read_readvariableop:
6savev2_batch_normalization_86_beta_read_readvariableopA
=savev2_batch_normalization_86_moving_mean_read_readvariableopE
Asavev2_batch_normalization_86_moving_variance_read_readvariableop/
+savev2_conv2d_75_kernel_read_readvariableop-
)savev2_conv2d_75_bias_read_readvariableop;
7savev2_batch_normalization_87_gamma_read_readvariableop:
6savev2_batch_normalization_87_beta_read_readvariableopA
=savev2_batch_normalization_87_moving_mean_read_readvariableopE
Asavev2_batch_normalization_87_moving_variance_read_readvariableop/
+savev2_conv2d_76_kernel_read_readvariableop-
)savev2_conv2d_76_bias_read_readvariableop;
7savev2_batch_normalization_88_gamma_read_readvariableop:
6savev2_batch_normalization_88_beta_read_readvariableopA
=savev2_batch_normalization_88_moving_mean_read_readvariableopE
Asavev2_batch_normalization_88_moving_variance_read_readvariableop/
+savev2_conv2d_77_kernel_read_readvariableop-
)savev2_conv2d_77_bias_read_readvariableop;
7savev2_batch_normalization_89_gamma_read_readvariableop:
6savev2_batch_normalization_89_beta_read_readvariableopA
=savev2_batch_normalization_89_moving_mean_read_readvariableopE
Asavev2_batch_normalization_89_moving_variance_read_readvariableop;
7savev2_batch_normalization_90_gamma_read_readvariableop:
6savev2_batch_normalization_90_beta_read_readvariableopA
=savev2_batch_normalization_90_moving_mean_read_readvariableopE
Asavev2_batch_normalization_90_moving_variance_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_72_kernel_read_readvariableop)savev2_conv2d_72_bias_read_readvariableop7savev2_batch_normalization_84_gamma_read_readvariableop6savev2_batch_normalization_84_beta_read_readvariableop=savev2_batch_normalization_84_moving_mean_read_readvariableopAsavev2_batch_normalization_84_moving_variance_read_readvariableop+savev2_conv2d_73_kernel_read_readvariableop)savev2_conv2d_73_bias_read_readvariableop7savev2_batch_normalization_85_gamma_read_readvariableop6savev2_batch_normalization_85_beta_read_readvariableop=savev2_batch_normalization_85_moving_mean_read_readvariableopAsavev2_batch_normalization_85_moving_variance_read_readvariableop+savev2_conv2d_74_kernel_read_readvariableop)savev2_conv2d_74_bias_read_readvariableop7savev2_batch_normalization_86_gamma_read_readvariableop6savev2_batch_normalization_86_beta_read_readvariableop=savev2_batch_normalization_86_moving_mean_read_readvariableopAsavev2_batch_normalization_86_moving_variance_read_readvariableop+savev2_conv2d_75_kernel_read_readvariableop)savev2_conv2d_75_bias_read_readvariableop7savev2_batch_normalization_87_gamma_read_readvariableop6savev2_batch_normalization_87_beta_read_readvariableop=savev2_batch_normalization_87_moving_mean_read_readvariableopAsavev2_batch_normalization_87_moving_variance_read_readvariableop+savev2_conv2d_76_kernel_read_readvariableop)savev2_conv2d_76_bias_read_readvariableop7savev2_batch_normalization_88_gamma_read_readvariableop6savev2_batch_normalization_88_beta_read_readvariableop=savev2_batch_normalization_88_moving_mean_read_readvariableopAsavev2_batch_normalization_88_moving_variance_read_readvariableop+savev2_conv2d_77_kernel_read_readvariableop)savev2_conv2d_77_bias_read_readvariableop7savev2_batch_normalization_89_gamma_read_readvariableop6savev2_batch_normalization_89_beta_read_readvariableop=savev2_batch_normalization_89_moving_mean_read_readvariableopAsavev2_batch_normalization_89_moving_variance_read_readvariableop7savev2_batch_normalization_90_gamma_read_readvariableop6savev2_batch_normalization_90_beta_read_readvariableop=savev2_batch_normalization_90_moving_mean_read_readvariableopAsavev2_batch_normalization_90_moving_variance_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797580

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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_795111

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
�
G
+__inference_flatten_12_layer_call_fn_797635

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
F__inference_flatten_12_layer_call_and_return_conditional_losses_7954282
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
D__inference_dense_12_layer_call_and_return_conditional_losses_795447

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
�
�
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797406

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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_795368

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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_794892

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
7__inference_batch_normalization_90_layer_call_fn_797624

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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_7953862
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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797304

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
�

*__inference_conv2d_72_layer_call_fn_796628

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
E__inference_conv2d_72_layer_call_and_return_conditional_losses_7947572
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
7__inference_batch_normalization_88_layer_call_fn_797348

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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_7952122
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

a
E__inference_lambda_12_layer_call_and_return_conditional_losses_795474

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
E__inference_conv2d_73_layer_call_and_return_conditional_losses_796767

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
7__inference_batch_normalization_85_layer_call_fn_796904

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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_7949102
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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_794511

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
��
�&
I__inference_sequential_12_layer_call_and_return_conditional_losses_796267

inputs,
(conv2d_72_conv2d_readvariableop_resource-
)conv2d_72_biasadd_readvariableop_resource2
.batch_normalization_84_readvariableop_resource4
0batch_normalization_84_readvariableop_1_resourceC
?batch_normalization_84_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_84_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_73_conv2d_readvariableop_resource-
)conv2d_73_biasadd_readvariableop_resource2
.batch_normalization_85_readvariableop_resource4
0batch_normalization_85_readvariableop_1_resourceC
?batch_normalization_85_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_85_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_74_conv2d_readvariableop_resource-
)conv2d_74_biasadd_readvariableop_resource2
.batch_normalization_86_readvariableop_resource4
0batch_normalization_86_readvariableop_1_resourceC
?batch_normalization_86_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_86_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_75_conv2d_readvariableop_resource-
)conv2d_75_biasadd_readvariableop_resource2
.batch_normalization_87_readvariableop_resource4
0batch_normalization_87_readvariableop_1_resourceC
?batch_normalization_87_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_87_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_76_conv2d_readvariableop_resource-
)conv2d_76_biasadd_readvariableop_resource2
.batch_normalization_88_readvariableop_resource4
0batch_normalization_88_readvariableop_1_resourceC
?batch_normalization_88_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_88_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_77_conv2d_readvariableop_resource-
)conv2d_77_biasadd_readvariableop_resource2
.batch_normalization_89_readvariableop_resource4
0batch_normalization_89_readvariableop_1_resourceC
?batch_normalization_89_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_89_fusedbatchnormv3_readvariableop_1_resource2
.batch_normalization_90_readvariableop_resource4
0batch_normalization_90_readvariableop_1_resourceC
?batch_normalization_90_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_90_fusedbatchnormv3_readvariableop_1_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity��%batch_normalization_84/AssignNewValue�'batch_normalization_84/AssignNewValue_1�6batch_normalization_84/FusedBatchNormV3/ReadVariableOp�8batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_84/ReadVariableOp�'batch_normalization_84/ReadVariableOp_1�%batch_normalization_85/AssignNewValue�'batch_normalization_85/AssignNewValue_1�6batch_normalization_85/FusedBatchNormV3/ReadVariableOp�8batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_85/ReadVariableOp�'batch_normalization_85/ReadVariableOp_1�%batch_normalization_86/AssignNewValue�'batch_normalization_86/AssignNewValue_1�6batch_normalization_86/FusedBatchNormV3/ReadVariableOp�8batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_86/ReadVariableOp�'batch_normalization_86/ReadVariableOp_1�%batch_normalization_87/AssignNewValue�'batch_normalization_87/AssignNewValue_1�6batch_normalization_87/FusedBatchNormV3/ReadVariableOp�8batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_87/ReadVariableOp�'batch_normalization_87/ReadVariableOp_1�%batch_normalization_88/AssignNewValue�'batch_normalization_88/AssignNewValue_1�6batch_normalization_88/FusedBatchNormV3/ReadVariableOp�8batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_88/ReadVariableOp�'batch_normalization_88/ReadVariableOp_1�%batch_normalization_89/AssignNewValue�'batch_normalization_89/AssignNewValue_1�6batch_normalization_89/FusedBatchNormV3/ReadVariableOp�8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_89/ReadVariableOp�'batch_normalization_89/ReadVariableOp_1�%batch_normalization_90/AssignNewValue�'batch_normalization_90/AssignNewValue_1�6batch_normalization_90/FusedBatchNormV3/ReadVariableOp�8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_90/ReadVariableOp�'batch_normalization_90/ReadVariableOp_1� conv2d_72/BiasAdd/ReadVariableOp�conv2d_72/Conv2D/ReadVariableOp� conv2d_73/BiasAdd/ReadVariableOp�conv2d_73/Conv2D/ReadVariableOp� conv2d_74/BiasAdd/ReadVariableOp�conv2d_74/Conv2D/ReadVariableOp� conv2d_75/BiasAdd/ReadVariableOp�conv2d_75/Conv2D/ReadVariableOp� conv2d_76/BiasAdd/ReadVariableOp�conv2d_76/Conv2D/ReadVariableOp� conv2d_77/BiasAdd/ReadVariableOp�conv2d_77/Conv2D/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�
conv2d_72/Conv2D/ReadVariableOpReadVariableOp(conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_72/Conv2D/ReadVariableOp�
conv2d_72/Conv2DConv2Dinputs'conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_72/Conv2D�
 conv2d_72/BiasAdd/ReadVariableOpReadVariableOp)conv2d_72_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_72/BiasAdd/ReadVariableOp�
conv2d_72/BiasAddBiasAddconv2d_72/Conv2D:output:0(conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_72/BiasAdd~
conv2d_72/ReluReluconv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_72/Relu�
%batch_normalization_84/ReadVariableOpReadVariableOp.batch_normalization_84_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_84/ReadVariableOp�
'batch_normalization_84/ReadVariableOp_1ReadVariableOp0batch_normalization_84_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_84/ReadVariableOp_1�
6batch_normalization_84/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_84_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_84/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_84_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_84/FusedBatchNormV3FusedBatchNormV3conv2d_72/Relu:activations:0-batch_normalization_84/ReadVariableOp:value:0/batch_normalization_84/ReadVariableOp_1:value:0>batch_normalization_84/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_84/FusedBatchNormV3�
%batch_normalization_84/AssignNewValueAssignVariableOp?batch_normalization_84_fusedbatchnormv3_readvariableop_resource4batch_normalization_84/FusedBatchNormV3:batch_mean:07^batch_normalization_84/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_84/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_84/AssignNewValue�
'batch_normalization_84/AssignNewValue_1AssignVariableOpAbatch_normalization_84_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_84/FusedBatchNormV3:batch_variance:09^batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_84/AssignNewValue_1�
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_73/Conv2D/ReadVariableOp�
conv2d_73/Conv2DConv2D+batch_normalization_84/FusedBatchNormV3:y:0'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv2d_73/Conv2D�
 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_73/BiasAdd/ReadVariableOp�
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv2d_73/BiasAdd~
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
conv2d_73/Relu�
%batch_normalization_85/ReadVariableOpReadVariableOp.batch_normalization_85_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_85/ReadVariableOp�
'batch_normalization_85/ReadVariableOp_1ReadVariableOp0batch_normalization_85_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_85/ReadVariableOp_1�
6batch_normalization_85/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_85_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_85/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_85_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_85/FusedBatchNormV3FusedBatchNormV3conv2d_73/Relu:activations:0-batch_normalization_85/ReadVariableOp:value:0/batch_normalization_85/ReadVariableOp_1:value:0>batch_normalization_85/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_85/FusedBatchNormV3�
%batch_normalization_85/AssignNewValueAssignVariableOp?batch_normalization_85_fusedbatchnormv3_readvariableop_resource4batch_normalization_85/FusedBatchNormV3:batch_mean:07^batch_normalization_85/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_85/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_85/AssignNewValue�
'batch_normalization_85/AssignNewValue_1AssignVariableOpAbatch_normalization_85_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_85/FusedBatchNormV3:batch_variance:09^batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_85/AssignNewValue_1�
max_pooling2d_36/MaxPoolMaxPool+batch_normalization_85/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_36/MaxPool�
conv2d_74/Conv2D/ReadVariableOpReadVariableOp(conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_74/Conv2D/ReadVariableOp�
conv2d_74/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_74/Conv2D�
 conv2d_74/BiasAdd/ReadVariableOpReadVariableOp)conv2d_74_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_74/BiasAdd/ReadVariableOp�
conv2d_74/BiasAddBiasAddconv2d_74/Conv2D:output:0(conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_74/BiasAdd~
conv2d_74/ReluReluconv2d_74/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_74/Relu�
%batch_normalization_86/ReadVariableOpReadVariableOp.batch_normalization_86_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_86/ReadVariableOp�
'batch_normalization_86/ReadVariableOp_1ReadVariableOp0batch_normalization_86_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_86/ReadVariableOp_1�
6batch_normalization_86/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_86_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_86/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_86_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_86/FusedBatchNormV3FusedBatchNormV3conv2d_74/Relu:activations:0-batch_normalization_86/ReadVariableOp:value:0/batch_normalization_86/ReadVariableOp_1:value:0>batch_normalization_86/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_86/FusedBatchNormV3�
%batch_normalization_86/AssignNewValueAssignVariableOp?batch_normalization_86_fusedbatchnormv3_readvariableop_resource4batch_normalization_86/FusedBatchNormV3:batch_mean:07^batch_normalization_86/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_86/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_86/AssignNewValue�
'batch_normalization_86/AssignNewValue_1AssignVariableOpAbatch_normalization_86_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_86/FusedBatchNormV3:batch_variance:09^batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_86/AssignNewValue_1�
conv2d_75/Conv2D/ReadVariableOpReadVariableOp(conv2d_75_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_75/Conv2D/ReadVariableOp�
conv2d_75/Conv2DConv2D+batch_normalization_86/FusedBatchNormV3:y:0'conv2d_75/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_75/Conv2D�
 conv2d_75/BiasAdd/ReadVariableOpReadVariableOp)conv2d_75_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_75/BiasAdd/ReadVariableOp�
conv2d_75/BiasAddBiasAddconv2d_75/Conv2D:output:0(conv2d_75/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_75/BiasAdd~
conv2d_75/ReluReluconv2d_75/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_75/Relu�
%batch_normalization_87/ReadVariableOpReadVariableOp.batch_normalization_87_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_87/ReadVariableOp�
'batch_normalization_87/ReadVariableOp_1ReadVariableOp0batch_normalization_87_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_87/ReadVariableOp_1�
6batch_normalization_87/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_87_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_87/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_87_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_87/FusedBatchNormV3FusedBatchNormV3conv2d_75/Relu:activations:0-batch_normalization_87/ReadVariableOp:value:0/batch_normalization_87/ReadVariableOp_1:value:0>batch_normalization_87/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_87/FusedBatchNormV3�
%batch_normalization_87/AssignNewValueAssignVariableOp?batch_normalization_87_fusedbatchnormv3_readvariableop_resource4batch_normalization_87/FusedBatchNormV3:batch_mean:07^batch_normalization_87/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_87/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_87/AssignNewValue�
'batch_normalization_87/AssignNewValue_1AssignVariableOpAbatch_normalization_87_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_87/FusedBatchNormV3:batch_variance:09^batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_87/AssignNewValue_1�
max_pooling2d_37/MaxPoolMaxPool+batch_normalization_87/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_37/MaxPool�
conv2d_76/Conv2D/ReadVariableOpReadVariableOp(conv2d_76_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_76/Conv2D/ReadVariableOp�
conv2d_76/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_76/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_76/Conv2D�
 conv2d_76/BiasAdd/ReadVariableOpReadVariableOp)conv2d_76_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_76/BiasAdd/ReadVariableOp�
conv2d_76/BiasAddBiasAddconv2d_76/Conv2D:output:0(conv2d_76/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_76/BiasAdd
conv2d_76/ReluReluconv2d_76/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_76/Relu�
%batch_normalization_88/ReadVariableOpReadVariableOp.batch_normalization_88_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_88/ReadVariableOp�
'batch_normalization_88/ReadVariableOp_1ReadVariableOp0batch_normalization_88_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_88/ReadVariableOp_1�
6batch_normalization_88/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_88_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_88/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_88_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_88/FusedBatchNormV3FusedBatchNormV3conv2d_76/Relu:activations:0-batch_normalization_88/ReadVariableOp:value:0/batch_normalization_88/ReadVariableOp_1:value:0>batch_normalization_88/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_88/FusedBatchNormV3�
%batch_normalization_88/AssignNewValueAssignVariableOp?batch_normalization_88_fusedbatchnormv3_readvariableop_resource4batch_normalization_88/FusedBatchNormV3:batch_mean:07^batch_normalization_88/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_88/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_88/AssignNewValue�
'batch_normalization_88/AssignNewValue_1AssignVariableOpAbatch_normalization_88_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_88/FusedBatchNormV3:batch_variance:09^batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_88/AssignNewValue_1�
conv2d_77/Conv2D/ReadVariableOpReadVariableOp(conv2d_77_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_77/Conv2D/ReadVariableOp�
conv2d_77/Conv2DConv2D+batch_normalization_88/FusedBatchNormV3:y:0'conv2d_77/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_77/Conv2D�
 conv2d_77/BiasAdd/ReadVariableOpReadVariableOp)conv2d_77_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_77/BiasAdd/ReadVariableOp�
conv2d_77/BiasAddBiasAddconv2d_77/Conv2D:output:0(conv2d_77/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_77/BiasAdd
conv2d_77/ReluReluconv2d_77/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_77/Relu�
%batch_normalization_89/ReadVariableOpReadVariableOp.batch_normalization_89_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_89/ReadVariableOp�
'batch_normalization_89/ReadVariableOp_1ReadVariableOp0batch_normalization_89_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_89/ReadVariableOp_1�
6batch_normalization_89/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_89_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_89/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_89_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_89/FusedBatchNormV3FusedBatchNormV3conv2d_77/Relu:activations:0-batch_normalization_89/ReadVariableOp:value:0/batch_normalization_89/ReadVariableOp_1:value:0>batch_normalization_89/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_89/FusedBatchNormV3�
%batch_normalization_89/AssignNewValueAssignVariableOp?batch_normalization_89_fusedbatchnormv3_readvariableop_resource4batch_normalization_89/FusedBatchNormV3:batch_mean:07^batch_normalization_89/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_89/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_89/AssignNewValue�
'batch_normalization_89/AssignNewValue_1AssignVariableOpAbatch_normalization_89_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_89/FusedBatchNormV3:batch_variance:09^batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_89/AssignNewValue_1�
max_pooling2d_38/MaxPoolMaxPool+batch_normalization_89/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_38/MaxPool�
%batch_normalization_90/ReadVariableOpReadVariableOp.batch_normalization_90_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%batch_normalization_90/ReadVariableOp�
'batch_normalization_90/ReadVariableOp_1ReadVariableOp0batch_normalization_90_readvariableop_1_resource*
_output_shapes	
:�*
dtype02)
'batch_normalization_90/ReadVariableOp_1�
6batch_normalization_90/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_90_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_90/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_90_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02:
8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_90/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_38/MaxPool:output:0-batch_normalization_90/ReadVariableOp:value:0/batch_normalization_90/ReadVariableOp_1:value:0>batch_normalization_90/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2)
'batch_normalization_90/FusedBatchNormV3�
%batch_normalization_90/AssignNewValueAssignVariableOp?batch_normalization_90_fusedbatchnormv3_readvariableop_resource4batch_normalization_90/FusedBatchNormV3:batch_mean:07^batch_normalization_90/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_90/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_90/AssignNewValue�
'batch_normalization_90/AssignNewValue_1AssignVariableOpAbatch_normalization_90_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_90/FusedBatchNormV3:batch_variance:09^batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_90/AssignNewValue_1u
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_12/Const�
flatten_12/ReshapeReshape+batch_normalization_90/FusedBatchNormV3:y:0flatten_12/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_12/Reshape�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulflatten_12/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAdd}
dense_12/SigmoidSigmoiddense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_12/Sigmoid�
lambda_12/l2_normalize/SquareSquaredense_12/Sigmoid:y:0*
T0*(
_output_shapes
:����������2
lambda_12/l2_normalize/Square�
,lambda_12/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,lambda_12/l2_normalize/Sum/reduction_indices�
lambda_12/l2_normalize/SumSum!lambda_12/l2_normalize/Square:y:05lambda_12/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2
lambda_12/l2_normalize/Sum�
 lambda_12/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+2"
 lambda_12/l2_normalize/Maximum/y�
lambda_12/l2_normalize/MaximumMaximum#lambda_12/l2_normalize/Sum:output:0)lambda_12/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2 
lambda_12/l2_normalize/Maximum�
lambda_12/l2_normalize/RsqrtRsqrt"lambda_12/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2
lambda_12/l2_normalize/Rsqrt�
lambda_12/l2_normalizeMuldense_12/Sigmoid:y:0 lambda_12/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2
lambda_12/l2_normalize�
IdentityIdentitylambda_12/l2_normalize:z:0&^batch_normalization_84/AssignNewValue(^batch_normalization_84/AssignNewValue_17^batch_normalization_84/FusedBatchNormV3/ReadVariableOp9^batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_84/ReadVariableOp(^batch_normalization_84/ReadVariableOp_1&^batch_normalization_85/AssignNewValue(^batch_normalization_85/AssignNewValue_17^batch_normalization_85/FusedBatchNormV3/ReadVariableOp9^batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_85/ReadVariableOp(^batch_normalization_85/ReadVariableOp_1&^batch_normalization_86/AssignNewValue(^batch_normalization_86/AssignNewValue_17^batch_normalization_86/FusedBatchNormV3/ReadVariableOp9^batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_86/ReadVariableOp(^batch_normalization_86/ReadVariableOp_1&^batch_normalization_87/AssignNewValue(^batch_normalization_87/AssignNewValue_17^batch_normalization_87/FusedBatchNormV3/ReadVariableOp9^batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_87/ReadVariableOp(^batch_normalization_87/ReadVariableOp_1&^batch_normalization_88/AssignNewValue(^batch_normalization_88/AssignNewValue_17^batch_normalization_88/FusedBatchNormV3/ReadVariableOp9^batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_88/ReadVariableOp(^batch_normalization_88/ReadVariableOp_1&^batch_normalization_89/AssignNewValue(^batch_normalization_89/AssignNewValue_17^batch_normalization_89/FusedBatchNormV3/ReadVariableOp9^batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_89/ReadVariableOp(^batch_normalization_89/ReadVariableOp_1&^batch_normalization_90/AssignNewValue(^batch_normalization_90/AssignNewValue_17^batch_normalization_90/FusedBatchNormV3/ReadVariableOp9^batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_90/ReadVariableOp(^batch_normalization_90/ReadVariableOp_1!^conv2d_72/BiasAdd/ReadVariableOp ^conv2d_72/Conv2D/ReadVariableOp!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp!^conv2d_74/BiasAdd/ReadVariableOp ^conv2d_74/Conv2D/ReadVariableOp!^conv2d_75/BiasAdd/ReadVariableOp ^conv2d_75/Conv2D/ReadVariableOp!^conv2d_76/BiasAdd/ReadVariableOp ^conv2d_76/Conv2D/ReadVariableOp!^conv2d_77/BiasAdd/ReadVariableOp ^conv2d_77/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_84/AssignNewValue%batch_normalization_84/AssignNewValue2R
'batch_normalization_84/AssignNewValue_1'batch_normalization_84/AssignNewValue_12p
6batch_normalization_84/FusedBatchNormV3/ReadVariableOp6batch_normalization_84/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_84/FusedBatchNormV3/ReadVariableOp_18batch_normalization_84/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_84/ReadVariableOp%batch_normalization_84/ReadVariableOp2R
'batch_normalization_84/ReadVariableOp_1'batch_normalization_84/ReadVariableOp_12N
%batch_normalization_85/AssignNewValue%batch_normalization_85/AssignNewValue2R
'batch_normalization_85/AssignNewValue_1'batch_normalization_85/AssignNewValue_12p
6batch_normalization_85/FusedBatchNormV3/ReadVariableOp6batch_normalization_85/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_85/FusedBatchNormV3/ReadVariableOp_18batch_normalization_85/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_85/ReadVariableOp%batch_normalization_85/ReadVariableOp2R
'batch_normalization_85/ReadVariableOp_1'batch_normalization_85/ReadVariableOp_12N
%batch_normalization_86/AssignNewValue%batch_normalization_86/AssignNewValue2R
'batch_normalization_86/AssignNewValue_1'batch_normalization_86/AssignNewValue_12p
6batch_normalization_86/FusedBatchNormV3/ReadVariableOp6batch_normalization_86/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_86/FusedBatchNormV3/ReadVariableOp_18batch_normalization_86/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_86/ReadVariableOp%batch_normalization_86/ReadVariableOp2R
'batch_normalization_86/ReadVariableOp_1'batch_normalization_86/ReadVariableOp_12N
%batch_normalization_87/AssignNewValue%batch_normalization_87/AssignNewValue2R
'batch_normalization_87/AssignNewValue_1'batch_normalization_87/AssignNewValue_12p
6batch_normalization_87/FusedBatchNormV3/ReadVariableOp6batch_normalization_87/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_87/FusedBatchNormV3/ReadVariableOp_18batch_normalization_87/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_87/ReadVariableOp%batch_normalization_87/ReadVariableOp2R
'batch_normalization_87/ReadVariableOp_1'batch_normalization_87/ReadVariableOp_12N
%batch_normalization_88/AssignNewValue%batch_normalization_88/AssignNewValue2R
'batch_normalization_88/AssignNewValue_1'batch_normalization_88/AssignNewValue_12p
6batch_normalization_88/FusedBatchNormV3/ReadVariableOp6batch_normalization_88/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_88/FusedBatchNormV3/ReadVariableOp_18batch_normalization_88/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_88/ReadVariableOp%batch_normalization_88/ReadVariableOp2R
'batch_normalization_88/ReadVariableOp_1'batch_normalization_88/ReadVariableOp_12N
%batch_normalization_89/AssignNewValue%batch_normalization_89/AssignNewValue2R
'batch_normalization_89/AssignNewValue_1'batch_normalization_89/AssignNewValue_12p
6batch_normalization_89/FusedBatchNormV3/ReadVariableOp6batch_normalization_89/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_18batch_normalization_89/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_89/ReadVariableOp%batch_normalization_89/ReadVariableOp2R
'batch_normalization_89/ReadVariableOp_1'batch_normalization_89/ReadVariableOp_12N
%batch_normalization_90/AssignNewValue%batch_normalization_90/AssignNewValue2R
'batch_normalization_90/AssignNewValue_1'batch_normalization_90/AssignNewValue_12p
6batch_normalization_90/FusedBatchNormV3/ReadVariableOp6batch_normalization_90/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_18batch_normalization_90/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_90/ReadVariableOp%batch_normalization_90/ReadVariableOp2R
'batch_normalization_90/ReadVariableOp_1'batch_normalization_90/ReadVariableOp_12D
 conv2d_72/BiasAdd/ReadVariableOp conv2d_72/BiasAdd/ReadVariableOp2B
conv2d_72/Conv2D/ReadVariableOpconv2d_72/Conv2D/ReadVariableOp2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp2D
 conv2d_74/BiasAdd/ReadVariableOp conv2d_74/BiasAdd/ReadVariableOp2B
conv2d_74/Conv2D/ReadVariableOpconv2d_74/Conv2D/ReadVariableOp2D
 conv2d_75/BiasAdd/ReadVariableOp conv2d_75/BiasAdd/ReadVariableOp2B
conv2d_75/Conv2D/ReadVariableOpconv2d_75/Conv2D/ReadVariableOp2D
 conv2d_76/BiasAdd/ReadVariableOp conv2d_76/BiasAdd/ReadVariableOp2B
conv2d_76/Conv2D/ReadVariableOpconv2d_76/Conv2D/ReadVariableOp2D
 conv2d_77/BiasAdd/ReadVariableOp conv2d_77/BiasAdd/ReadVariableOp2B
conv2d_77/Conv2D/ReadVariableOpconv2d_77/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796648

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
�
M
1__inference_max_pooling2d_38_layer_call_fn_794638

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
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_7946322
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
�
b
F__inference_flatten_12_layer_call_and_return_conditional_losses_795428

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
�i
�
I__inference_sequential_12_layer_call_and_return_conditional_losses_795716

inputs
conv2d_72_795612
conv2d_72_795614!
batch_normalization_84_795617!
batch_normalization_84_795619!
batch_normalization_84_795621!
batch_normalization_84_795623
conv2d_73_795626
conv2d_73_795628!
batch_normalization_85_795631!
batch_normalization_85_795633!
batch_normalization_85_795635!
batch_normalization_85_795637
conv2d_74_795641
conv2d_74_795643!
batch_normalization_86_795646!
batch_normalization_86_795648!
batch_normalization_86_795650!
batch_normalization_86_795652
conv2d_75_795655
conv2d_75_795657!
batch_normalization_87_795660!
batch_normalization_87_795662!
batch_normalization_87_795664!
batch_normalization_87_795666
conv2d_76_795670
conv2d_76_795672!
batch_normalization_88_795675!
batch_normalization_88_795677!
batch_normalization_88_795679!
batch_normalization_88_795681
conv2d_77_795684
conv2d_77_795686!
batch_normalization_89_795689!
batch_normalization_89_795691!
batch_normalization_89_795693!
batch_normalization_89_795695!
batch_normalization_90_795699!
batch_normalization_90_795701!
batch_normalization_90_795703!
batch_normalization_90_795705
dense_12_795709
dense_12_795711
identity��.batch_normalization_84/StatefulPartitionedCall�.batch_normalization_85/StatefulPartitionedCall�.batch_normalization_86/StatefulPartitionedCall�.batch_normalization_87/StatefulPartitionedCall�.batch_normalization_88/StatefulPartitionedCall�.batch_normalization_89/StatefulPartitionedCall�.batch_normalization_90/StatefulPartitionedCall�!conv2d_72/StatefulPartitionedCall�!conv2d_73/StatefulPartitionedCall�!conv2d_74/StatefulPartitionedCall�!conv2d_75/StatefulPartitionedCall�!conv2d_76/StatefulPartitionedCall�!conv2d_77/StatefulPartitionedCall� dense_12/StatefulPartitionedCall�
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_72_795612conv2d_72_795614*
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
E__inference_conv2d_72_layer_call_and_return_conditional_losses_7947572#
!conv2d_72/StatefulPartitionedCall�
.batch_normalization_84/StatefulPartitionedCallStatefulPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0batch_normalization_84_795617batch_normalization_84_795619batch_normalization_84_795621batch_normalization_84_795623*
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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_79479220
.batch_normalization_84/StatefulPartitionedCall�
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_84/StatefulPartitionedCall:output:0conv2d_73_795626conv2d_73_795628*
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
E__inference_conv2d_73_layer_call_and_return_conditional_losses_7948572#
!conv2d_73/StatefulPartitionedCall�
.batch_normalization_85/StatefulPartitionedCallStatefulPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0batch_normalization_85_795631batch_normalization_85_795633batch_normalization_85_795635batch_normalization_85_795637*
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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_79489220
.batch_normalization_85/StatefulPartitionedCall�
 max_pooling2d_36/PartitionedCallPartitionedCall7batch_normalization_85/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_7941922"
 max_pooling2d_36/PartitionedCall�
!conv2d_74/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_74_795641conv2d_74_795643*
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
E__inference_conv2d_74_layer_call_and_return_conditional_losses_7949582#
!conv2d_74/StatefulPartitionedCall�
.batch_normalization_86/StatefulPartitionedCallStatefulPartitionedCall*conv2d_74/StatefulPartitionedCall:output:0batch_normalization_86_795646batch_normalization_86_795648batch_normalization_86_795650batch_normalization_86_795652*
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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_79499320
.batch_normalization_86/StatefulPartitionedCall�
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_86/StatefulPartitionedCall:output:0conv2d_75_795655conv2d_75_795657*
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
E__inference_conv2d_75_layer_call_and_return_conditional_losses_7950582#
!conv2d_75/StatefulPartitionedCall�
.batch_normalization_87/StatefulPartitionedCallStatefulPartitionedCall*conv2d_75/StatefulPartitionedCall:output:0batch_normalization_87_795660batch_normalization_87_795662batch_normalization_87_795664batch_normalization_87_795666*
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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_79509320
.batch_normalization_87/StatefulPartitionedCall�
 max_pooling2d_37/PartitionedCallPartitionedCall7batch_normalization_87/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_7944122"
 max_pooling2d_37/PartitionedCall�
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_76_795670conv2d_76_795672*
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
E__inference_conv2d_76_layer_call_and_return_conditional_losses_7951592#
!conv2d_76/StatefulPartitionedCall�
.batch_normalization_88/StatefulPartitionedCallStatefulPartitionedCall*conv2d_76/StatefulPartitionedCall:output:0batch_normalization_88_795675batch_normalization_88_795677batch_normalization_88_795679batch_normalization_88_795681*
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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_79519420
.batch_normalization_88/StatefulPartitionedCall�
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_88/StatefulPartitionedCall:output:0conv2d_77_795684conv2d_77_795686*
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
E__inference_conv2d_77_layer_call_and_return_conditional_losses_7952592#
!conv2d_77/StatefulPartitionedCall�
.batch_normalization_89/StatefulPartitionedCallStatefulPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0batch_normalization_89_795689batch_normalization_89_795691batch_normalization_89_795693batch_normalization_89_795695*
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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_79529420
.batch_normalization_89/StatefulPartitionedCall�
 max_pooling2d_38/PartitionedCallPartitionedCall7batch_normalization_89/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_7946322"
 max_pooling2d_38/PartitionedCall�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_38/PartitionedCall:output:0batch_normalization_90_795699batch_normalization_90_795701batch_normalization_90_795703batch_normalization_90_795705*
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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_79536820
.batch_normalization_90/StatefulPartitionedCall�
flatten_12/PartitionedCallPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0*
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
F__inference_flatten_12_layer_call_and_return_conditional_losses_7954282
flatten_12/PartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0dense_12_795709dense_12_795711*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_7954472"
 dense_12/StatefulPartitionedCall�
lambda_12/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
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
E__inference_lambda_12_layer_call_and_return_conditional_losses_7954742
lambda_12/PartitionedCall�
IdentityIdentity"lambda_12/PartitionedCall:output:0/^batch_normalization_84/StatefulPartitionedCall/^batch_normalization_85/StatefulPartitionedCall/^batch_normalization_86/StatefulPartitionedCall/^batch_normalization_87/StatefulPartitionedCall/^batch_normalization_88/StatefulPartitionedCall/^batch_normalization_89/StatefulPartitionedCall/^batch_normalization_90/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall"^conv2d_74/StatefulPartitionedCall"^conv2d_75/StatefulPartitionedCall"^conv2d_76/StatefulPartitionedCall"^conv2d_77/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_84/StatefulPartitionedCall.batch_normalization_84/StatefulPartitionedCall2`
.batch_normalization_85/StatefulPartitionedCall.batch_normalization_85/StatefulPartitionedCall2`
.batch_normalization_86/StatefulPartitionedCall.batch_normalization_86/StatefulPartitionedCall2`
.batch_normalization_87/StatefulPartitionedCall.batch_normalization_87/StatefulPartitionedCall2`
.batch_normalization_88/StatefulPartitionedCall.batch_normalization_88/StatefulPartitionedCall2`
.batch_normalization_89/StatefulPartitionedCall.batch_normalization_89/StatefulPartitionedCall2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2F
!conv2d_74/StatefulPartitionedCall!conv2d_74/StatefulPartitionedCall2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
��
�*
!__inference__wrapped_model_793978
conv2d_72_input:
6sequential_12_conv2d_72_conv2d_readvariableop_resource;
7sequential_12_conv2d_72_biasadd_readvariableop_resource@
<sequential_12_batch_normalization_84_readvariableop_resourceB
>sequential_12_batch_normalization_84_readvariableop_1_resourceQ
Msequential_12_batch_normalization_84_fusedbatchnormv3_readvariableop_resourceS
Osequential_12_batch_normalization_84_fusedbatchnormv3_readvariableop_1_resource:
6sequential_12_conv2d_73_conv2d_readvariableop_resource;
7sequential_12_conv2d_73_biasadd_readvariableop_resource@
<sequential_12_batch_normalization_85_readvariableop_resourceB
>sequential_12_batch_normalization_85_readvariableop_1_resourceQ
Msequential_12_batch_normalization_85_fusedbatchnormv3_readvariableop_resourceS
Osequential_12_batch_normalization_85_fusedbatchnormv3_readvariableop_1_resource:
6sequential_12_conv2d_74_conv2d_readvariableop_resource;
7sequential_12_conv2d_74_biasadd_readvariableop_resource@
<sequential_12_batch_normalization_86_readvariableop_resourceB
>sequential_12_batch_normalization_86_readvariableop_1_resourceQ
Msequential_12_batch_normalization_86_fusedbatchnormv3_readvariableop_resourceS
Osequential_12_batch_normalization_86_fusedbatchnormv3_readvariableop_1_resource:
6sequential_12_conv2d_75_conv2d_readvariableop_resource;
7sequential_12_conv2d_75_biasadd_readvariableop_resource@
<sequential_12_batch_normalization_87_readvariableop_resourceB
>sequential_12_batch_normalization_87_readvariableop_1_resourceQ
Msequential_12_batch_normalization_87_fusedbatchnormv3_readvariableop_resourceS
Osequential_12_batch_normalization_87_fusedbatchnormv3_readvariableop_1_resource:
6sequential_12_conv2d_76_conv2d_readvariableop_resource;
7sequential_12_conv2d_76_biasadd_readvariableop_resource@
<sequential_12_batch_normalization_88_readvariableop_resourceB
>sequential_12_batch_normalization_88_readvariableop_1_resourceQ
Msequential_12_batch_normalization_88_fusedbatchnormv3_readvariableop_resourceS
Osequential_12_batch_normalization_88_fusedbatchnormv3_readvariableop_1_resource:
6sequential_12_conv2d_77_conv2d_readvariableop_resource;
7sequential_12_conv2d_77_biasadd_readvariableop_resource@
<sequential_12_batch_normalization_89_readvariableop_resourceB
>sequential_12_batch_normalization_89_readvariableop_1_resourceQ
Msequential_12_batch_normalization_89_fusedbatchnormv3_readvariableop_resourceS
Osequential_12_batch_normalization_89_fusedbatchnormv3_readvariableop_1_resource@
<sequential_12_batch_normalization_90_readvariableop_resourceB
>sequential_12_batch_normalization_90_readvariableop_1_resourceQ
Msequential_12_batch_normalization_90_fusedbatchnormv3_readvariableop_resourceS
Osequential_12_batch_normalization_90_fusedbatchnormv3_readvariableop_1_resource9
5sequential_12_dense_12_matmul_readvariableop_resource:
6sequential_12_dense_12_biasadd_readvariableop_resource
identity��Dsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp�Fsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1�3sequential_12/batch_normalization_84/ReadVariableOp�5sequential_12/batch_normalization_84/ReadVariableOp_1�Dsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp�Fsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1�3sequential_12/batch_normalization_85/ReadVariableOp�5sequential_12/batch_normalization_85/ReadVariableOp_1�Dsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp�Fsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1�3sequential_12/batch_normalization_86/ReadVariableOp�5sequential_12/batch_normalization_86/ReadVariableOp_1�Dsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp�Fsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1�3sequential_12/batch_normalization_87/ReadVariableOp�5sequential_12/batch_normalization_87/ReadVariableOp_1�Dsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp�Fsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1�3sequential_12/batch_normalization_88/ReadVariableOp�5sequential_12/batch_normalization_88/ReadVariableOp_1�Dsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp�Fsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1�3sequential_12/batch_normalization_89/ReadVariableOp�5sequential_12/batch_normalization_89/ReadVariableOp_1�Dsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp�Fsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1�3sequential_12/batch_normalization_90/ReadVariableOp�5sequential_12/batch_normalization_90/ReadVariableOp_1�.sequential_12/conv2d_72/BiasAdd/ReadVariableOp�-sequential_12/conv2d_72/Conv2D/ReadVariableOp�.sequential_12/conv2d_73/BiasAdd/ReadVariableOp�-sequential_12/conv2d_73/Conv2D/ReadVariableOp�.sequential_12/conv2d_74/BiasAdd/ReadVariableOp�-sequential_12/conv2d_74/Conv2D/ReadVariableOp�.sequential_12/conv2d_75/BiasAdd/ReadVariableOp�-sequential_12/conv2d_75/Conv2D/ReadVariableOp�.sequential_12/conv2d_76/BiasAdd/ReadVariableOp�-sequential_12/conv2d_76/Conv2D/ReadVariableOp�.sequential_12/conv2d_77/BiasAdd/ReadVariableOp�-sequential_12/conv2d_77/Conv2D/ReadVariableOp�-sequential_12/dense_12/BiasAdd/ReadVariableOp�,sequential_12/dense_12/MatMul/ReadVariableOp�
-sequential_12/conv2d_72/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_12/conv2d_72/Conv2D/ReadVariableOp�
sequential_12/conv2d_72/Conv2DConv2Dconv2d_72_input5sequential_12/conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2 
sequential_12/conv2d_72/Conv2D�
.sequential_12/conv2d_72/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_72_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_12/conv2d_72/BiasAdd/ReadVariableOp�
sequential_12/conv2d_72/BiasAddBiasAdd'sequential_12/conv2d_72/Conv2D:output:06sequential_12/conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2!
sequential_12/conv2d_72/BiasAdd�
sequential_12/conv2d_72/ReluRelu(sequential_12/conv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
sequential_12/conv2d_72/Relu�
3sequential_12/batch_normalization_84/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_84_readvariableop_resource*
_output_shapes
: *
dtype025
3sequential_12/batch_normalization_84/ReadVariableOp�
5sequential_12/batch_normalization_84/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_84_readvariableop_1_resource*
_output_shapes
: *
dtype027
5sequential_12/batch_normalization_84/ReadVariableOp_1�
Dsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_84_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp�
Fsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_84_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1�
5sequential_12/batch_normalization_84/FusedBatchNormV3FusedBatchNormV3*sequential_12/conv2d_72/Relu:activations:0;sequential_12/batch_normalization_84/ReadVariableOp:value:0=sequential_12/batch_normalization_84/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 27
5sequential_12/batch_normalization_84/FusedBatchNormV3�
-sequential_12/conv2d_73/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02/
-sequential_12/conv2d_73/Conv2D/ReadVariableOp�
sequential_12/conv2d_73/Conv2DConv2D9sequential_12/batch_normalization_84/FusedBatchNormV3:y:05sequential_12/conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2 
sequential_12/conv2d_73/Conv2D�
.sequential_12/conv2d_73/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_12/conv2d_73/BiasAdd/ReadVariableOp�
sequential_12/conv2d_73/BiasAddBiasAdd'sequential_12/conv2d_73/Conv2D:output:06sequential_12/conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2!
sequential_12/conv2d_73/BiasAdd�
sequential_12/conv2d_73/ReluRelu(sequential_12/conv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:���������   2
sequential_12/conv2d_73/Relu�
3sequential_12/batch_normalization_85/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_85_readvariableop_resource*
_output_shapes
: *
dtype025
3sequential_12/batch_normalization_85/ReadVariableOp�
5sequential_12/batch_normalization_85/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_85_readvariableop_1_resource*
_output_shapes
: *
dtype027
5sequential_12/batch_normalization_85/ReadVariableOp_1�
Dsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_85_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp�
Fsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_85_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1�
5sequential_12/batch_normalization_85/FusedBatchNormV3FusedBatchNormV3*sequential_12/conv2d_73/Relu:activations:0;sequential_12/batch_normalization_85/ReadVariableOp:value:0=sequential_12/batch_normalization_85/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 27
5sequential_12/batch_normalization_85/FusedBatchNormV3�
&sequential_12/max_pooling2d_36/MaxPoolMaxPool9sequential_12/batch_normalization_85/FusedBatchNormV3:y:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2(
&sequential_12/max_pooling2d_36/MaxPool�
-sequential_12/conv2d_74/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_12/conv2d_74/Conv2D/ReadVariableOp�
sequential_12/conv2d_74/Conv2DConv2D/sequential_12/max_pooling2d_36/MaxPool:output:05sequential_12/conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2 
sequential_12/conv2d_74/Conv2D�
.sequential_12/conv2d_74/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_74_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_12/conv2d_74/BiasAdd/ReadVariableOp�
sequential_12/conv2d_74/BiasAddBiasAdd'sequential_12/conv2d_74/Conv2D:output:06sequential_12/conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2!
sequential_12/conv2d_74/BiasAdd�
sequential_12/conv2d_74/ReluRelu(sequential_12/conv2d_74/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential_12/conv2d_74/Relu�
3sequential_12/batch_normalization_86/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_86_readvariableop_resource*
_output_shapes
:@*
dtype025
3sequential_12/batch_normalization_86/ReadVariableOp�
5sequential_12/batch_normalization_86/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_86_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5sequential_12/batch_normalization_86/ReadVariableOp_1�
Dsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_86_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp�
Fsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_86_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1�
5sequential_12/batch_normalization_86/FusedBatchNormV3FusedBatchNormV3*sequential_12/conv2d_74/Relu:activations:0;sequential_12/batch_normalization_86/ReadVariableOp:value:0=sequential_12/batch_normalization_86/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5sequential_12/batch_normalization_86/FusedBatchNormV3�
-sequential_12/conv2d_75/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_75_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_12/conv2d_75/Conv2D/ReadVariableOp�
sequential_12/conv2d_75/Conv2DConv2D9sequential_12/batch_normalization_86/FusedBatchNormV3:y:05sequential_12/conv2d_75/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2 
sequential_12/conv2d_75/Conv2D�
.sequential_12/conv2d_75/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_75_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_12/conv2d_75/BiasAdd/ReadVariableOp�
sequential_12/conv2d_75/BiasAddBiasAdd'sequential_12/conv2d_75/Conv2D:output:06sequential_12/conv2d_75/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2!
sequential_12/conv2d_75/BiasAdd�
sequential_12/conv2d_75/ReluRelu(sequential_12/conv2d_75/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential_12/conv2d_75/Relu�
3sequential_12/batch_normalization_87/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_87_readvariableop_resource*
_output_shapes
:@*
dtype025
3sequential_12/batch_normalization_87/ReadVariableOp�
5sequential_12/batch_normalization_87/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_87_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5sequential_12/batch_normalization_87/ReadVariableOp_1�
Dsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_87_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp�
Fsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_87_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1�
5sequential_12/batch_normalization_87/FusedBatchNormV3FusedBatchNormV3*sequential_12/conv2d_75/Relu:activations:0;sequential_12/batch_normalization_87/ReadVariableOp:value:0=sequential_12/batch_normalization_87/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5sequential_12/batch_normalization_87/FusedBatchNormV3�
&sequential_12/max_pooling2d_37/MaxPoolMaxPool9sequential_12/batch_normalization_87/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2(
&sequential_12/max_pooling2d_37/MaxPool�
-sequential_12/conv2d_76/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_76_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02/
-sequential_12/conv2d_76/Conv2D/ReadVariableOp�
sequential_12/conv2d_76/Conv2DConv2D/sequential_12/max_pooling2d_37/MaxPool:output:05sequential_12/conv2d_76/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2 
sequential_12/conv2d_76/Conv2D�
.sequential_12/conv2d_76/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_76_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.sequential_12/conv2d_76/BiasAdd/ReadVariableOp�
sequential_12/conv2d_76/BiasAddBiasAdd'sequential_12/conv2d_76/Conv2D:output:06sequential_12/conv2d_76/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2!
sequential_12/conv2d_76/BiasAdd�
sequential_12/conv2d_76/ReluRelu(sequential_12/conv2d_76/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_12/conv2d_76/Relu�
3sequential_12/batch_normalization_88/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_88_readvariableop_resource*
_output_shapes	
:�*
dtype025
3sequential_12/batch_normalization_88/ReadVariableOp�
5sequential_12/batch_normalization_88/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_88_readvariableop_1_resource*
_output_shapes	
:�*
dtype027
5sequential_12/batch_normalization_88/ReadVariableOp_1�
Dsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_88_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp�
Fsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_88_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02H
Fsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1�
5sequential_12/batch_normalization_88/FusedBatchNormV3FusedBatchNormV3*sequential_12/conv2d_76/Relu:activations:0;sequential_12/batch_normalization_88/ReadVariableOp:value:0=sequential_12/batch_normalization_88/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 27
5sequential_12/batch_normalization_88/FusedBatchNormV3�
-sequential_12/conv2d_77/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_77_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02/
-sequential_12/conv2d_77/Conv2D/ReadVariableOp�
sequential_12/conv2d_77/Conv2DConv2D9sequential_12/batch_normalization_88/FusedBatchNormV3:y:05sequential_12/conv2d_77/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2 
sequential_12/conv2d_77/Conv2D�
.sequential_12/conv2d_77/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_77_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.sequential_12/conv2d_77/BiasAdd/ReadVariableOp�
sequential_12/conv2d_77/BiasAddBiasAdd'sequential_12/conv2d_77/Conv2D:output:06sequential_12/conv2d_77/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2!
sequential_12/conv2d_77/BiasAdd�
sequential_12/conv2d_77/ReluRelu(sequential_12/conv2d_77/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential_12/conv2d_77/Relu�
3sequential_12/batch_normalization_89/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_89_readvariableop_resource*
_output_shapes	
:�*
dtype025
3sequential_12/batch_normalization_89/ReadVariableOp�
5sequential_12/batch_normalization_89/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_89_readvariableop_1_resource*
_output_shapes	
:�*
dtype027
5sequential_12/batch_normalization_89/ReadVariableOp_1�
Dsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_89_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp�
Fsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_89_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02H
Fsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1�
5sequential_12/batch_normalization_89/FusedBatchNormV3FusedBatchNormV3*sequential_12/conv2d_77/Relu:activations:0;sequential_12/batch_normalization_89/ReadVariableOp:value:0=sequential_12/batch_normalization_89/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 27
5sequential_12/batch_normalization_89/FusedBatchNormV3�
&sequential_12/max_pooling2d_38/MaxPoolMaxPool9sequential_12/batch_normalization_89/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2(
&sequential_12/max_pooling2d_38/MaxPool�
3sequential_12/batch_normalization_90/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_90_readvariableop_resource*
_output_shapes	
:�*
dtype025
3sequential_12/batch_normalization_90/ReadVariableOp�
5sequential_12/batch_normalization_90/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_90_readvariableop_1_resource*
_output_shapes	
:�*
dtype027
5sequential_12/batch_normalization_90/ReadVariableOp_1�
Dsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_90_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp�
Fsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_90_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02H
Fsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1�
5sequential_12/batch_normalization_90/FusedBatchNormV3FusedBatchNormV3/sequential_12/max_pooling2d_38/MaxPool:output:0;sequential_12/batch_normalization_90/ReadVariableOp:value:0=sequential_12/batch_normalization_90/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 27
5sequential_12/batch_normalization_90/FusedBatchNormV3�
sequential_12/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2 
sequential_12/flatten_12/Const�
 sequential_12/flatten_12/ReshapeReshape9sequential_12/batch_normalization_90/FusedBatchNormV3:y:0'sequential_12/flatten_12/Const:output:0*
T0*(
_output_shapes
:����������2"
 sequential_12/flatten_12/Reshape�
,sequential_12/dense_12/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,sequential_12/dense_12/MatMul/ReadVariableOp�
sequential_12/dense_12/MatMulMatMul)sequential_12/flatten_12/Reshape:output:04sequential_12/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_12/dense_12/MatMul�
-sequential_12/dense_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_12/dense_12/BiasAdd/ReadVariableOp�
sequential_12/dense_12/BiasAddBiasAdd'sequential_12/dense_12/MatMul:product:05sequential_12/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_12/dense_12/BiasAdd�
sequential_12/dense_12/SigmoidSigmoid'sequential_12/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:����������2 
sequential_12/dense_12/Sigmoid�
+sequential_12/lambda_12/l2_normalize/SquareSquare"sequential_12/dense_12/Sigmoid:y:0*
T0*(
_output_shapes
:����������2-
+sequential_12/lambda_12/l2_normalize/Square�
:sequential_12/lambda_12/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_12/lambda_12/l2_normalize/Sum/reduction_indices�
(sequential_12/lambda_12/l2_normalize/SumSum/sequential_12/lambda_12/l2_normalize/Square:y:0Csequential_12/lambda_12/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2*
(sequential_12/lambda_12/l2_normalize/Sum�
.sequential_12/lambda_12/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+20
.sequential_12/lambda_12/l2_normalize/Maximum/y�
,sequential_12/lambda_12/l2_normalize/MaximumMaximum1sequential_12/lambda_12/l2_normalize/Sum:output:07sequential_12/lambda_12/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2.
,sequential_12/lambda_12/l2_normalize/Maximum�
*sequential_12/lambda_12/l2_normalize/RsqrtRsqrt0sequential_12/lambda_12/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2,
*sequential_12/lambda_12/l2_normalize/Rsqrt�
$sequential_12/lambda_12/l2_normalizeMul"sequential_12/dense_12/Sigmoid:y:0.sequential_12/lambda_12/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2&
$sequential_12/lambda_12/l2_normalize�
IdentityIdentity(sequential_12/lambda_12/l2_normalize:z:0E^sequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_84/ReadVariableOp6^sequential_12/batch_normalization_84/ReadVariableOp_1E^sequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_85/ReadVariableOp6^sequential_12/batch_normalization_85/ReadVariableOp_1E^sequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_86/ReadVariableOp6^sequential_12/batch_normalization_86/ReadVariableOp_1E^sequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_87/ReadVariableOp6^sequential_12/batch_normalization_87/ReadVariableOp_1E^sequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_88/ReadVariableOp6^sequential_12/batch_normalization_88/ReadVariableOp_1E^sequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_89/ReadVariableOp6^sequential_12/batch_normalization_89/ReadVariableOp_1E^sequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_90/ReadVariableOp6^sequential_12/batch_normalization_90/ReadVariableOp_1/^sequential_12/conv2d_72/BiasAdd/ReadVariableOp.^sequential_12/conv2d_72/Conv2D/ReadVariableOp/^sequential_12/conv2d_73/BiasAdd/ReadVariableOp.^sequential_12/conv2d_73/Conv2D/ReadVariableOp/^sequential_12/conv2d_74/BiasAdd/ReadVariableOp.^sequential_12/conv2d_74/Conv2D/ReadVariableOp/^sequential_12/conv2d_75/BiasAdd/ReadVariableOp.^sequential_12/conv2d_75/Conv2D/ReadVariableOp/^sequential_12/conv2d_76/BiasAdd/ReadVariableOp.^sequential_12/conv2d_76/Conv2D/ReadVariableOp/^sequential_12/conv2d_77/BiasAdd/ReadVariableOp.^sequential_12/conv2d_77/Conv2D/ReadVariableOp.^sequential_12/dense_12/BiasAdd/ReadVariableOp-^sequential_12/dense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������  ::::::::::::::::::::::::::::::::::::::::::2�
Dsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp2�
Fsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_84/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_84/ReadVariableOp3sequential_12/batch_normalization_84/ReadVariableOp2n
5sequential_12/batch_normalization_84/ReadVariableOp_15sequential_12/batch_normalization_84/ReadVariableOp_12�
Dsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp2�
Fsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_85/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_85/ReadVariableOp3sequential_12/batch_normalization_85/ReadVariableOp2n
5sequential_12/batch_normalization_85/ReadVariableOp_15sequential_12/batch_normalization_85/ReadVariableOp_12�
Dsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp2�
Fsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_86/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_86/ReadVariableOp3sequential_12/batch_normalization_86/ReadVariableOp2n
5sequential_12/batch_normalization_86/ReadVariableOp_15sequential_12/batch_normalization_86/ReadVariableOp_12�
Dsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp2�
Fsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_87/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_87/ReadVariableOp3sequential_12/batch_normalization_87/ReadVariableOp2n
5sequential_12/batch_normalization_87/ReadVariableOp_15sequential_12/batch_normalization_87/ReadVariableOp_12�
Dsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp2�
Fsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_88/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_88/ReadVariableOp3sequential_12/batch_normalization_88/ReadVariableOp2n
5sequential_12/batch_normalization_88/ReadVariableOp_15sequential_12/batch_normalization_88/ReadVariableOp_12�
Dsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp2�
Fsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_89/ReadVariableOp3sequential_12/batch_normalization_89/ReadVariableOp2n
5sequential_12/batch_normalization_89/ReadVariableOp_15sequential_12/batch_normalization_89/ReadVariableOp_12�
Dsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp2�
Fsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_90/ReadVariableOp3sequential_12/batch_normalization_90/ReadVariableOp2n
5sequential_12/batch_normalization_90/ReadVariableOp_15sequential_12/batch_normalization_90/ReadVariableOp_12`
.sequential_12/conv2d_72/BiasAdd/ReadVariableOp.sequential_12/conv2d_72/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_72/Conv2D/ReadVariableOp-sequential_12/conv2d_72/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_73/BiasAdd/ReadVariableOp.sequential_12/conv2d_73/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_73/Conv2D/ReadVariableOp-sequential_12/conv2d_73/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_74/BiasAdd/ReadVariableOp.sequential_12/conv2d_74/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_74/Conv2D/ReadVariableOp-sequential_12/conv2d_74/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_75/BiasAdd/ReadVariableOp.sequential_12/conv2d_75/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_75/Conv2D/ReadVariableOp-sequential_12/conv2d_75/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_76/BiasAdd/ReadVariableOp.sequential_12/conv2d_76/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_76/Conv2D/ReadVariableOp-sequential_12/conv2d_76/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_77/BiasAdd/ReadVariableOp.sequential_12/conv2d_77/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_77/Conv2D/ReadVariableOp-sequential_12/conv2d_77/Conv2D/ReadVariableOp2^
-sequential_12/dense_12/BiasAdd/ReadVariableOp-sequential_12/dense_12/BiasAdd/ReadVariableOp2\
,sequential_12/dense_12/MatMul/ReadVariableOp,sequential_12/dense_12/MatMul/ReadVariableOp:` \
/
_output_shapes
:���������  
)
_user_specified_nameconv2d_72_input
�

�
E__inference_conv2d_76_layer_call_and_return_conditional_losses_795159

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
�
�
7__inference_batch_normalization_86_layer_call_fn_796975

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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_7942602
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
~
)__inference_dense_12_layer_call_fn_797655

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
D__inference_dense_12_layer_call_and_return_conditional_losses_7954472
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
�
�
7__inference_batch_normalization_87_layer_call_fn_797136

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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_7943952
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
conv2d_72_input@
!serving_default_conv2d_72_input:0���������  >
	lambda_121
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
_tf_keras_sequential֣{"class_name": "Sequential", "name": "sequential_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_72_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_72", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_84", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_73", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_85", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_74", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_86", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_75", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_87", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_76", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_88", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_77", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_89", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_90", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Lambda", "config": {"name": "lambda_12", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPoe\nPGlweXRob24taW5wdXQtNS1kODA2ODAzY2Y1YTY+2gg8bGFtYmRhPhgAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_72_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_72", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_84", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_73", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_85", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_74", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_86", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_75", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_87", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_76", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_88", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_77", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_89", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_90", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Lambda", "config": {"name": "lambda_12", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPoe\nPGlweXRob24taW5wdXQtNS1kODA2ODAzY2Y1YTY+2gg8bGFtYmRhPhgAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}}}
�


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_72", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_72", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_84", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_84", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�	

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_73", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_73", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_85", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_85", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
�
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_36", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_74", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_74", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_86", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_86", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�	

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_75", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_75", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_87", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_87", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
�
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_37", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_76", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_76", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_88", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_88", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�	

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_77", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_77", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_89", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_89", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
�
{regularization_losses
|	variables
}trainable_variables
~	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_38", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
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
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_90", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_90", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 2048, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "lambda_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_12", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAUwAAAHMSAAAAdABqAWoCagN8AGQBZAKNAlMAKQNO6QEAAAAp\nAdoEYXhpcykE2gJ0ZtoFa2VyYXPaB2JhY2tlbmTaDGwyX25vcm1hbGl6ZSkB2gF4qQByCAAAAPoe\nPGlweXRob24taW5wdXQtNS1kODA2ODAzY2Y1YTY+2gg8bGFtYmRhPhgAAADzAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
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
*:( 2conv2d_72/kernel
: 2conv2d_72/bias
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
*:( 2batch_normalization_84/gamma
):' 2batch_normalization_84/beta
2:0  (2"batch_normalization_84/moving_mean
6:4  (2&batch_normalization_84/moving_variance
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
*:(  2conv2d_73/kernel
: 2conv2d_73/bias
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
*:( 2batch_normalization_85/gamma
):' 2batch_normalization_85/beta
2:0  (2"batch_normalization_85/moving_mean
6:4  (2&batch_normalization_85/moving_variance
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
*:( @2conv2d_74/kernel
:@2conv2d_74/bias
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
*:(@2batch_normalization_86/gamma
):'@2batch_normalization_86/beta
2:0@ (2"batch_normalization_86/moving_mean
6:4@ (2&batch_normalization_86/moving_variance
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
*:(@@2conv2d_75/kernel
:@2conv2d_75/bias
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
*:(@2batch_normalization_87/gamma
):'@2batch_normalization_87/beta
2:0@ (2"batch_normalization_87/moving_mean
6:4@ (2&batch_normalization_87/moving_variance
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
+:)@�2conv2d_76/kernel
:�2conv2d_76/bias
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
+:)�2batch_normalization_88/gamma
*:(�2batch_normalization_88/beta
3:1� (2"batch_normalization_88/moving_mean
7:5� (2&batch_normalization_88/moving_variance
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
,:*��2conv2d_77/kernel
:�2conv2d_77/bias
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
+:)�2batch_normalization_89/gamma
*:(�2batch_normalization_89/beta
3:1� (2"batch_normalization_89/moving_mean
7:5� (2&batch_normalization_89/moving_variance
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
+:)�2batch_normalization_90/gamma
*:(�2batch_normalization_90/beta
3:1� (2"batch_normalization_90/moving_mean
7:5� (2&batch_normalization_90/moving_variance
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
��2dense_12/kernel
:�2dense_12/bias
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_795499
I__inference_sequential_12_layer_call_and_return_conditional_losses_796267
I__inference_sequential_12_layer_call_and_return_conditional_losses_796430
I__inference_sequential_12_layer_call_and_return_conditional_losses_795606�
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
!__inference__wrapped_model_793978�
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
conv2d_72_input���������  
�2�
.__inference_sequential_12_layer_call_fn_796608
.__inference_sequential_12_layer_call_fn_796519
.__inference_sequential_12_layer_call_fn_795999
.__inference_sequential_12_layer_call_fn_795803�
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
E__inference_conv2d_72_layer_call_and_return_conditional_losses_796619�
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
*__inference_conv2d_72_layer_call_fn_796628�
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
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796648
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796666
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796730
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796712�
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
7__inference_batch_normalization_84_layer_call_fn_796743
7__inference_batch_normalization_84_layer_call_fn_796692
7__inference_batch_normalization_84_layer_call_fn_796679
7__inference_batch_normalization_84_layer_call_fn_796756�
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
E__inference_conv2d_73_layer_call_and_return_conditional_losses_796767�
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
*__inference_conv2d_73_layer_call_fn_796776�
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
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796878
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796860
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796814
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796796�
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
7__inference_batch_normalization_85_layer_call_fn_796904
7__inference_batch_normalization_85_layer_call_fn_796891
7__inference_batch_normalization_85_layer_call_fn_796827
7__inference_batch_normalization_85_layer_call_fn_796840�
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
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_794192�
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
1__inference_max_pooling2d_36_layer_call_fn_794198�
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
E__inference_conv2d_74_layer_call_and_return_conditional_losses_796915�
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
*__inference_conv2d_74_layer_call_fn_796924�
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
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_797008
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_797026
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_796944
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_796962�
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
7__inference_batch_normalization_86_layer_call_fn_797039
7__inference_batch_normalization_86_layer_call_fn_797052
7__inference_batch_normalization_86_layer_call_fn_796975
7__inference_batch_normalization_86_layer_call_fn_796988�
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
E__inference_conv2d_75_layer_call_and_return_conditional_losses_797063�
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
*__inference_conv2d_75_layer_call_fn_797072�
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
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797110
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797174
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797156
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797092�
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
7__inference_batch_normalization_87_layer_call_fn_797123
7__inference_batch_normalization_87_layer_call_fn_797136
7__inference_batch_normalization_87_layer_call_fn_797187
7__inference_batch_normalization_87_layer_call_fn_797200�
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
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_794412�
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
1__inference_max_pooling2d_37_layer_call_fn_794418�
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
E__inference_conv2d_76_layer_call_and_return_conditional_losses_797211�
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
*__inference_conv2d_76_layer_call_fn_797220�
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
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797258
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797240
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797304
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797322�
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
7__inference_batch_normalization_88_layer_call_fn_797284
7__inference_batch_normalization_88_layer_call_fn_797271
7__inference_batch_normalization_88_layer_call_fn_797348
7__inference_batch_normalization_88_layer_call_fn_797335�
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
E__inference_conv2d_77_layer_call_and_return_conditional_losses_797359�
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
*__inference_conv2d_77_layer_call_fn_797368�
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
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797470
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797452
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797406
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797388�
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
7__inference_batch_normalization_89_layer_call_fn_797432
7__inference_batch_normalization_89_layer_call_fn_797496
7__inference_batch_normalization_89_layer_call_fn_797483
7__inference_batch_normalization_89_layer_call_fn_797419�
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
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_794632�
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
1__inference_max_pooling2d_38_layer_call_fn_794638�
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
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797580
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797534
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797598
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797516�
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
7__inference_batch_normalization_90_layer_call_fn_797560
7__inference_batch_normalization_90_layer_call_fn_797611
7__inference_batch_normalization_90_layer_call_fn_797624
7__inference_batch_normalization_90_layer_call_fn_797547�
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
F__inference_flatten_12_layer_call_and_return_conditional_losses_797630�
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
+__inference_flatten_12_layer_call_fn_797635�
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
D__inference_dense_12_layer_call_and_return_conditional_losses_797646�
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
)__inference_dense_12_layer_call_fn_797655�
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
E__inference_lambda_12_layer_call_and_return_conditional_losses_797677
E__inference_lambda_12_layer_call_and_return_conditional_losses_797666�
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
*__inference_lambda_12_layer_call_fn_797687
*__inference_lambda_12_layer_call_fn_797682�
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
$__inference_signature_wrapper_796090conv2d_72_input"�
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
!__inference__wrapped_model_793978�0 !"#()/012;<BCDEJKQRST]^defglmstuv������@�=
6�3
1�.
conv2d_72_input���������  
� "6�3
1
	lambda_12$�!
	lambda_12�����������
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796648r !"#;�8
1�.
(�%
inputs���������   
p
� "-�*
#� 
0���������   
� �
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796666r !"#;�8
1�.
(�%
inputs���������   
p 
� "-�*
#� 
0���������   
� �
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796712� !"#M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_84_layer_call_and_return_conditional_losses_796730� !"#M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
7__inference_batch_normalization_84_layer_call_fn_796679e !"#;�8
1�.
(�%
inputs���������   
p
� " ����������   �
7__inference_batch_normalization_84_layer_call_fn_796692e !"#;�8
1�.
(�%
inputs���������   
p 
� " ����������   �
7__inference_batch_normalization_84_layer_call_fn_796743� !"#M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
7__inference_batch_normalization_84_layer_call_fn_796756� !"#M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796796�/012M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796814�/012M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796860r/012;�8
1�.
(�%
inputs���������   
p
� "-�*
#� 
0���������   
� �
R__inference_batch_normalization_85_layer_call_and_return_conditional_losses_796878r/012;�8
1�.
(�%
inputs���������   
p 
� "-�*
#� 
0���������   
� �
7__inference_batch_normalization_85_layer_call_fn_796827�/012M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
7__inference_batch_normalization_85_layer_call_fn_796840�/012M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
7__inference_batch_normalization_85_layer_call_fn_796891e/012;�8
1�.
(�%
inputs���������   
p
� " ����������   �
7__inference_batch_normalization_85_layer_call_fn_796904e/012;�8
1�.
(�%
inputs���������   
p 
� " ����������   �
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_796944�BCDEM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_796962�BCDEM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_797008rBCDE;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_86_layer_call_and_return_conditional_losses_797026rBCDE;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
7__inference_batch_normalization_86_layer_call_fn_796975�BCDEM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_86_layer_call_fn_796988�BCDEM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_86_layer_call_fn_797039eBCDE;�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_86_layer_call_fn_797052eBCDE;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797092�QRSTM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797110�QRSTM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797156rQRST;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_87_layer_call_and_return_conditional_losses_797174rQRST;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
7__inference_batch_normalization_87_layer_call_fn_797123�QRSTM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_87_layer_call_fn_797136�QRSTM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_87_layer_call_fn_797187eQRST;�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_87_layer_call_fn_797200eQRST;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797240�defgN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797258�defgN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797304tdefg<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
R__inference_batch_normalization_88_layer_call_and_return_conditional_losses_797322tdefg<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
7__inference_batch_normalization_88_layer_call_fn_797271�defgN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
7__inference_batch_normalization_88_layer_call_fn_797284�defgN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
7__inference_batch_normalization_88_layer_call_fn_797335gdefg<�9
2�/
)�&
inputs����������
p
� "!������������
7__inference_batch_normalization_88_layer_call_fn_797348gdefg<�9
2�/
)�&
inputs����������
p 
� "!������������
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797388�stuvN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797406�stuvN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797452tstuv<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_797470tstuv<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
7__inference_batch_normalization_89_layer_call_fn_797419�stuvN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
7__inference_batch_normalization_89_layer_call_fn_797432�stuvN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
7__inference_batch_normalization_89_layer_call_fn_797483gstuv<�9
2�/
)�&
inputs����������
p
� "!������������
7__inference_batch_normalization_89_layer_call_fn_797496gstuv<�9
2�/
)�&
inputs����������
p 
� "!������������
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797516�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797534�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797580x����<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_797598x����<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
7__inference_batch_normalization_90_layer_call_fn_797547�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
7__inference_batch_normalization_90_layer_call_fn_797560�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
7__inference_batch_normalization_90_layer_call_fn_797611k����<�9
2�/
)�&
inputs����������
p
� "!������������
7__inference_batch_normalization_90_layer_call_fn_797624k����<�9
2�/
)�&
inputs����������
p 
� "!������������
E__inference_conv2d_72_layer_call_and_return_conditional_losses_796619l7�4
-�*
(�%
inputs���������  
� "-�*
#� 
0���������   
� �
*__inference_conv2d_72_layer_call_fn_796628_7�4
-�*
(�%
inputs���������  
� " ����������   �
E__inference_conv2d_73_layer_call_and_return_conditional_losses_796767l()7�4
-�*
(�%
inputs���������   
� "-�*
#� 
0���������   
� �
*__inference_conv2d_73_layer_call_fn_796776_()7�4
-�*
(�%
inputs���������   
� " ����������   �
E__inference_conv2d_74_layer_call_and_return_conditional_losses_796915l;<7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
*__inference_conv2d_74_layer_call_fn_796924_;<7�4
-�*
(�%
inputs��������� 
� " ����������@�
E__inference_conv2d_75_layer_call_and_return_conditional_losses_797063lJK7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
*__inference_conv2d_75_layer_call_fn_797072_JK7�4
-�*
(�%
inputs���������@
� " ����������@�
E__inference_conv2d_76_layer_call_and_return_conditional_losses_797211m]^7�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
*__inference_conv2d_76_layer_call_fn_797220`]^7�4
-�*
(�%
inputs���������@
� "!������������
E__inference_conv2d_77_layer_call_and_return_conditional_losses_797359nlm8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
*__inference_conv2d_77_layer_call_fn_797368alm8�5
.�+
)�&
inputs����������
� "!������������
D__inference_dense_12_layer_call_and_return_conditional_losses_797646`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
)__inference_dense_12_layer_call_fn_797655S��0�-
&�#
!�
inputs����������
� "������������
F__inference_flatten_12_layer_call_and_return_conditional_losses_797630b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
+__inference_flatten_12_layer_call_fn_797635U8�5
.�+
)�&
inputs����������
� "������������
E__inference_lambda_12_layer_call_and_return_conditional_losses_797666b8�5
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
E__inference_lambda_12_layer_call_and_return_conditional_losses_797677b8�5
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
*__inference_lambda_12_layer_call_fn_797682U8�5
.�+
!�
inputs����������

 
p
� "������������
*__inference_lambda_12_layer_call_fn_797687U8�5
.�+
!�
inputs����������

 
p 
� "������������
L__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_794192�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_36_layer_call_fn_794198�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_794412�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_37_layer_call_fn_794418�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_794632�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_38_layer_call_fn_794638�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_sequential_12_layer_call_and_return_conditional_losses_795499�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_72_input���������  
p

 
� "&�#
�
0����������
� �
I__inference_sequential_12_layer_call_and_return_conditional_losses_795606�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_72_input���������  
p 

 
� "&�#
�
0����������
� �
I__inference_sequential_12_layer_call_and_return_conditional_losses_796267�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_796430�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
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
.__inference_sequential_12_layer_call_fn_795803�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_72_input���������  
p

 
� "������������
.__inference_sequential_12_layer_call_fn_795999�0 !"#()/012;<BCDEJKQRST]^defglmstuv������H�E
>�;
1�.
conv2d_72_input���������  
p 

 
� "������������
.__inference_sequential_12_layer_call_fn_796519�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
5�2
(�%
inputs���������  
p

 
� "������������
.__inference_sequential_12_layer_call_fn_796608�0 !"#()/012;<BCDEJKQRST]^defglmstuv������?�<
5�2
(�%
inputs���������  
p 

 
� "������������
$__inference_signature_wrapper_796090�0 !"#()/012;<BCDEJKQRST]^defglmstuv������S�P
� 
I�F
D
conv2d_72_input1�.
conv2d_72_input���������  "6�3
1
	lambda_12$�!
	lambda_12����������