// dg_1d_conservation_law_example.c
// 一维守恒律: du/dt + df(u)/dx = 0 的最小不连续DG方法实现
// 以线性对流方程为例: du/dt + a du/dx = 0
// 使用LGL节点，质量矩阵为对角阵，D矩阵实现导数，数值通量为迎风格式


//一维守恒律方程的DG方法求解（不使用FEniCS）
//使用numpy和scipy实现
//方程: du/dt + 2π du/dx = 0
//初始条件: u(x,0) = sin(x)
//左边界: u(0,t) = -sin(2πt)
//右边界: 自然边界条件
//解析解: u(x,t) = sin(x - 2πt)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>

#define N_ELEM 30      // 单元数
// 五阶LGL节点数
#define N_P 5          // 每单元LGL节点数（五阶）
#define N_T 1000    // 时间步数
#define DT 0.001       // 时间步长
#define X_MIN 0.0
#define X_MAX (2*M_PI) 
#define A (2*M_PI)     // 对流速度

// LGL节点和权重（五阶，标准区间[-1,1]）
const double lgl_x[N_P] = {-1.0, -0.654653670707977, 0.0, 0.654653670707977, 1.0};
const double lgl_w[N_P] = {0.1, 0.544444444444444, 0.711111111111111, 0.544444444444444, 0.1};

// Stiffness矩阵（五阶，标准区间[-1,1]，精确值）
const double S[N_P][N_P] = {
    {-0.5000,  0.6757, -0.2667,  0.1410, -0.0500},
    {-0.6757, -0.0000,  0.9505, -0.4158,  0.1410},
    { 0.2667, -0.9505, -0.0000,  0.9505, -0.2667},
    {-0.1410,  0.4158, -0.9505,  0.0000,  0.6757},
    { 0.0500, -0.1410,  0.2667, -0.6757,  0.5000}
};

// 质量矩阵（五阶，标准区间[-1,1]，精确质量矩阵，满阵）
const double M[N_P][N_P] = {
    {0.0889,  0.0259, -0.0296,  0.0259, -0.0111},
    {0.0259,  0.4840,  0.0691, -0.0605,  0.0259},
    {-0.0296, 0.0691,  0.6321,  0.0691, -0.0296},
    {0.0259, -0.0605,  0.0691,  0.4840,  0.0259},
    {-0.0111, 0.0259, -0.0296,  0.0259,  0.0889}
};

// 精确质量矩阵的逆（参考单元），物理单元缩放由 (2/h) 体现
const double Minv[N_P][N_P] = {
    {12.5000, -1.0714,  0.9375, -1.0714,  2.5000},
    {-1.0714,  2.2959, -0.4018,  0.4592, -1.0714},
    { 0.9375, -0.4018,  1.7578, -0.4018,  0.9375},
    {-1.0714,  0.4592, -0.4018,  2.2959, -1.0714},
    { 2.5000, -1.0714,  0.9375, -1.0714, 12.5000}
};

// 初始条件
void initial_condition(double u[N_ELEM][N_P], double *x) {
    for (int e = 0; e < N_ELEM; ++e)
        for (int p = 0; p < N_P; ++p) {
            double xx = x[e] + 0.5*(lgl_x[p]+1.0)*(x[e+1]-x[e]);
            u[e][p] =0*xx;//*sin(xx); // 初始条件为sin(x)
        }
}

// 迎风数值通量
double num_flux(double ul, double ur) {
    return A * ul; // 对流速度a>0，使用左状态值
}

// 解析解：u(x,t) = sin(x - 2πt)
double exact_solution(double x, double t) {
    return sin(x - 2*M_PI*t);
}



// 计算右端项
void compute_rhs(double u[N_ELEM][N_P], double *x, double current_time, double rhs[N_ELEM][N_P]) {
    // 初始化右端项
    for (int e = 0; e < N_ELEM; ++e)
        for (int p = 0; p < N_P; ++p)
            rhs[e][p] = 0.0;
    
    // 体积分项 + 数值通量项：分开各自乘以 Minv，更直观
    for (int e = 0; e < N_ELEM; ++e) {
        double h = x[e+1] - x[e];

        // 体积分：v = A * (S^T u)，rhs += (2/h) * Minv * v
        double v[N_P];
        for (int j = 0; j < N_P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N_P; ++k) sum += S[k][j] * u[e][k];
            v[j] = A * sum;
        }
        for (int i = 0; i < N_P; ++i) {
            double yi = 0.0;
            for (int j = 0; j < N_P; ++j) yi += Minv[i][j] * v[j];
            rhs[e][i] += (2.0/h) * yi;
        }

        // 右界面通量：g_right = [0, ..., 0, -f*]，rhs += (2/h) * Minv * g_right
        if (e < N_ELEM - 1) {
            double ul = u[e][N_P-1];
            double ur = u[e+1][0];
            double fstar = num_flux(ul, ur);
            double g_right[N_P] = {0};
            g_right[N_P-1] = fstar;
            for (int i = 0; i < N_P; ++i) {
                double yi = 0.0;
                for (int j = 0; j < N_P; ++j) yi += Minv[i][j] * g_right[j];
                rhs[e][i] -= (2.0/h) * yi;
            }
        }

        // 左界面通量：g_left = [f*, 0, ..., 0]，rhs += (2/h) * Minv * g_left
        {
            double g_left_vec[N_P] = {0};
            if (e == 0) {
                double g_left = -sin(2*M_PI*current_time);
                double fstar = num_flux(g_left, u[e][0]);
                g_left_vec[0] = fstar;
            } else {
                double ul = u[e-1][N_P-1];
                double ur = u[e][0];
                double fstar = num_flux(ul, ur);
                g_left_vec[0] = fstar;
            }
            for (int i = 0; i < N_P; ++i) {
                double yi = 0.0;
                for (int j = 0; j < N_P; ++j) yi += Minv[i][j] * g_left_vec[j];
                rhs[e][i] += (2.0/h) * yi;
            }
        }
    }
}

// DG主循环：RK4时间推进
void dg_step_rk4(double u[N_ELEM][N_P], double *x, double dt, double current_time) {
    double k1[N_ELEM][N_P], k2[N_ELEM][N_P], k3[N_ELEM][N_P], k4[N_ELEM][N_P];
    double u_temp[N_ELEM][N_P];
    
    // 第一步：k1 = f(t, u)
    compute_rhs(u, x, current_time, k1);
    
    // 第二步：k2 = f(t + 0.5*dt, u + 0.5*dt*k1)
    for (int e = 0; e < N_ELEM; ++e)
        for (int p = 0; p < N_P; ++p)
            u_temp[e][p] = u[e][p] + 0.5 * dt * k1[e][p];
    compute_rhs(u_temp, x, current_time + 0.5 * dt, k2);
    
    // 第三步：k3 = f(t + 0.5*dt, u + 0.5*dt*k2)
    for (int e = 0; e < N_ELEM; ++e)
        for (int p = 0; p < N_P; ++p)
            u_temp[e][p] = u[e][p] + 0.5 * dt * k2[e][p];
    compute_rhs(u_temp, x, current_time + 0.5 * dt, k3);
    
    // 第四步：k4 = f(t + dt, u + dt*k3)
    for (int e = 0; e < N_ELEM; ++e)
        for (int p = 0; p < N_P; ++p)
            u_temp[e][p] = u[e][p] + dt * k3[e][p];
    compute_rhs(u_temp, x, current_time + dt, k4);
    
    // 最终更新：u = u + dt * (k1/6 + k2/3 + k3/3 + k4/6)
    for (int e = 0; e < N_ELEM; ++e)
        for (int p = 0; p < N_P; ++p)
            u[e][p] += dt * (k1[e][p]/6.0 + k2[e][p]/3.0 + k3[e][p]/3.0 + k4[e][p]/6.0);
}

int main() {
    double x[N_ELEM+1];
    for (int e = 0; e <= N_ELEM; ++e)
        x[e] = X_MIN + (X_MAX - X_MIN) * e / N_ELEM;
    double u[N_ELEM][N_P];
    initial_condition(u, x);
    // 帧输出目录
    mkdir("frames", 0777);
    // 时间推进
    for (int n = 0; n < N_T; ++n) {
        double current_time = n * DT;
        dg_step_rk4(u, x, DT, current_time);
        
        // 在RK4之后检查，比较 t = current_time + DT 的解
        double next_time = current_time + DT;
        double u_exact_0 = exact_solution(0.0, next_time);  // x=0处的精确解
        double u_exact_20 = exact_solution(x[20] + 0.5*(lgl_x[0]+1.0)*(x[21]-x[20]), next_time);  // 第20单元第0个节点的精确解
        
        printf("时间步 %d, t = %.3f -> %.3f\n", n, current_time, next_time);
        printf("  左边界: 数值解 = %f, 精确解 = %f, 误差 = %f\n", 
                u[0][0], u_exact_0, fabs(u[0][0] - u_exact_0));
        printf("  中间点: 数值解 = %f, 精确解 = %f, 误差 = %f\n", 
                u[20][0], u_exact_20, fabs(u[20][0] - u_exact_20));

        // 每10步输出一次：三列 x u u_exact（对应 next_time），仅用全局网格 x 节点
        if ((n % 10) == 0) {
            char path[128];
            snprintf(path, sizeof(path), "frames/step_%05d.txt", n);
            FILE *fout = fopen(path, "w");
            if (fout) {
                fprintf(fout, "# x u_numerical u_exact\n");
                for (int e = 0; e < N_ELEM-5; ++e) {
                    double xx = x[e];
                    double u_val = u[e][0];
                    double u_ex = exact_solution(xx, next_time);
                    fprintf(fout, "%f %f %f\n", xx, u_val, u_ex);
                }
                // {
                //     double xx = x[N_ELEM];
                //     double u_val = u[N_ELEM-1][N_P-1];
                //     double u_ex = exact_solution(xx, next_time);
                //     fprintf(fout, "%f %f %f\n", xx, u_val, u_ex);
                // }
                fclose(fout);
            }
        }

    }
    // // 计算误差并输出结果
    // double max_error = 0.0;
    // double l2_error = 0.0;
    // double total_weight = 0.0;
    
    // FILE *fp = fopen("dg_1d_result.txt", "w");
    // FILE *fp_error = fopen("dg_1d_error.txt", "w");
    
    // fprintf(fp, "# x u_numerical u_exact error\n");
    // fprintf(fp_error, "# t L2_error Max_error\n");
    
    // for (int e = 0; e < N_ELEM; ++e) {
    //     for (int p = 0; p < N_P; ++p) {
    //         double xx = x[e] + 0.5*(lgl_x[p]+1.0)*(x[e+1]-x[e]);
    //         double u_exact = exact_solution(xx, N_T * DT);
    //         double error = fabs(u[e][p] - u_exact);
            
    //         // 更新最大误差
    //         if (error > max_error) max_error = error;
            
    //         // 计算L2误差（使用LGL权重）
    //         double h = x[e+1] - x[e];
    //         double weight = 0.5 * h * lgl_w[p];
    //         l2_error += weight * error * error;
    //         total_weight += weight;
            
    //         fprintf(fp, "%f %f %f %f\n", xx, u[e][p], u_exact, error);
    //     }
    // }
    
    // // 计算L2误差
    // l2_error = sqrt(l2_error / total_weight);
    
    // fprintf(fp_error, "%f %e %e\n", N_T * DT, l2_error, max_error);
    // fclose(fp);
    // fclose(fp_error);
    
    // printf("DG计算完成，结果已保存到dg_1d_result.txt\n");
    // printf("误差分析：\n");
    // printf("L2误差: %e\n", l2_error);
    // printf("最大误差: %e\n", max_error);
    // printf("误差详情已保存到dg_1d_error.txt\n");
    return 0;
} 