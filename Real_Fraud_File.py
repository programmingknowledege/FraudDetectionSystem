# Fraud_Detection_file
from tkinter import *
import pandas as pd

from tkinter import simpledialog

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# from sklearn.model_selection import cross_val_score


class FraudDetection:
    def __init__(self, root):
        self.f = Frame(root, width=1000, height=1000, bg="goldenrod")
        self.ff = Frame(self.f, width=300, height=1000, bg="light goldenrod")
        self.loadDataset = Button(self.ff, text="LoadDataset", height=2, width=30, command=self.loaddataset,
                                  bg="goldenrod")
        self.loadDataset.place(x=40, y=20)
        self.Headpoints = Button(self.ff, text="Head Points", height=2, width=30, command=self.headpoint,
                                 bg="goldenrod")
        self.Headpoints.place(x=40, y=90)
        self.splitDataset = Button(self.ff, text="splitDataset", height=2, width=30, command=self.splitt,
                                   bg="goldenrod")
        self.splitDataset.place(x=40, y=160)
        self.checkAlgorithm = Button(self.ff, text="Spot and check algorithm ", height=2, width=30,
                                     command=self.spotcheck, bg="goldenrod")
        self.checkAlgorithm.place(x=40, y=230)
        self.CompareAlgorithms = Button(self.ff, height=1, width=40,
                                        bg="goldenrod")
        self.CompareAlgorithms.place(x=10, y=300)
        self.validation = Button(self.ff, text="Make predictions on validation dataset", height=2, width=30,
                                 command=self.classificationn, bg="goldenrod")
        self.validation.place(x=40, y=350)
        self.testset = Button(self.ff, text="Test Set", height=2, width=30, command=self.testsetprediction,
                              bg="goldenrod")
        self.testset.place(x=40, y=420)
        self.CompareAlgorithms = Button(self.ff, height=1, width=40,
                                        bg="goldenrod")
        self.CompareAlgorithms.place(x=10, y=480)

        self.realtimefraudd = Button(self.ff, height=2, width=30, text="Real Time Fraud",
                                     bg="goldenrod", command=self.realtimefraud)
        self.realtimefraudd.place(x=40, y=520)
        self.ff.pack(side=LEFT)
        self.fff = Frame(self.f, width=650, height=1000, bg="light goldenrod")
        self.scrollbar = Scrollbar(self.fff)
        self.scrollbar.pack(side=RIGHT, fill=BOTH)
        self.ffff = Text(self.fff, height=900, yscrollcommand=self.scrollbar.set)
        self.ffff.pack(side=LEFT, padx=20, pady=20)

        self.scrollbar.config(command=self.ffff.yview)

        self.df = pd.read_csv("train.csv", sep="|",
                              header=0)

        self.array = self.df.values
        self.array = self.df.values
        self.X = self.array[:, 0:9]
        self.Y = self.array[:, 9]
        self.validation_size = 0.20
        self.seed = 7
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = model_selection.train_test_split(self.X,
                                                                                                            self.Y,
                                                                                                            test_size=self.validation_size,
                                                                                                            random_state=0)

        self.fff.pack()
        self.f.pack()

    def loaddataset(self):
        self.df = pd.read_csv("train.csv", sep="|",
                              header=0)
        self.ffff.insert(END, self.df)

    def headpoint(self):
        self.ffff.delete('1.0', END)
        self.answer = simpledialog.askstring("Input", "Enter the value", parent=root)
        if self.answer is not None:
            self.ffff.insert(END, self.df[0:(int(self.answer))])
        else:
            print("Enter the value")

    def splitt(self):

        self.ffff.delete('1.0', END)
        self.ffff.insert(END, self.X.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.Y.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.X_train.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.Y_train.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.X_validation.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.Y_validation.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.X_train[0:5])

    def spotcheck(self):
        self.seed = 7
        self.scoring = 'accuracy'
        self.ffff.delete('1.0', END)
        self.models = []
        self.models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        self.models.append(('KNN', KNeighborsClassifier()))
        self.models.append(('CART', DecisionTreeClassifier()))
        # models.append(('NB', GaussianNB()))
        self.models.append(('SVM', SVC(gamma='auto')))
        self.models.append(('RF', RandomForestClassifier(n_estimators=10)))
        self.models.append(('ADA', AdaBoostClassifier(n_estimators=100)))

        # evaluate each model in turn
        self.results = []
        self.names = []
        for name, model in self.models:
            self.kfold = model_selection.KFold(n_splits=10)
            self.cv_results = model_selection.cross_val_score(model, self.X_train, self.Y_train, cv=self.kfold,
                                                              scoring=self.scoring)
            self.results.append(self.cv_results)
            self.names.append(name)
            self.msg = "%s: %f (%f)" % (name, self.cv_results.mean(), self.cv_results.std())
            self.ffff.insert(END, self.msg)
            self.ffff.insert(END, "\n")

    def classificationn(self):
        self.ada = AdaBoostClassifier(n_estimators=100)
        self.ada.fit(self.X_train, self.Y_train)
        self.predictions = self.ada.predict(self.X_validation)
        self.ffff.delete('1.0', END)
        self.ffff.insert(END, accuracy_score(self.Y_validation, self.predictions))
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, confusion_matrix(self.Y_validation, self.predictions))
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, classification_report(self.Y_validation, self.predictions))
        self.ffff.insert(END, "\n")

    def testsetprediction(self):
        self.ffff.delete('1.0', END)
        self.testset = pd.read_csv("test.csv",
                                   sep="|", header=0)
        self.ffff.insert(END, self.testset.shape)
        self.ffff.insert(END, "\n")
        self.Y_pred = self.ada.predict(self.testset)
        for i in range(0, len(self.Y_pred)):
            self.ffff.insert(END, self.Y_pred[i])
            self.ffff.insert(END, "\n")

    def realtimefraud(self):
        newWindow = Toplevel(root, height="1000", width="1000")
        # sets the title of the
        # Toplevel widget
        newWindow.title("Real Time Fraud Detection")
        self.f = Frame(newWindow, height="1000", width="300", bg="goldenrod")
        self.f.pack(side=LEFT)

        self.trustlevel = Label(self.f, text="trustlevel", width=30, bg="goldenrod", height=2)
        self.trustlevel.pack(padx=20, pady=8)

        self.totalScanTimeInSeconds = Label(self.f, text="totalScanTimeInSeconds", bg="goldenrod", width=30,
                                            height=2)
        self.totalScanTimeInSeconds.pack(padx=20, pady=8)

        self.grandTotal = Label(self.f, text="grandTotal", bg="goldenrod", width=30, height=2)
        self.grandTotal.pack(padx=20, pady=8)

        self.lineItemVoids = Label(self.f, text="lineItemVoids", bg="goldenrod", width=30, height=2)
        self.lineItemVoids.pack(padx=20, pady=8)

        self.scansWithoutRegistration = Label(self.f, text="scansWithoutRegistration", bg="goldenrod", width=30,
                                              height=2)
        self.scansWithoutRegistration.pack(padx=20, pady=8)

        self.quantityModification = Label(self.f, text="quantityModification", bg="goldenrod", width=30, height=2)
        self.quantityModification.pack(padx=20, pady=8)

        self.scannedLineItemsPerSecond = Label(self.f, text="scannedLineItemsPerSecond", bg="goldenrod", width=30,
                                               height=2)
        self.scannedLineItemsPerSecond.pack(padx=20, pady=8)

        self.valuePerSecond = Label(self.f, text="valuePerSecond", bg="goldenrod", width=30, height=2)
        self.valuePerSecond.pack(padx=20, pady=8)

        self.lineItemVoidsPerPosition = Label(self.f, bg="goldenrod", text="lineItemVoidsPerPosition", width=30,
                                              height=2)
        self.lineItemVoidsPerPosition.pack(padx=20, pady=8)

        self.fraud = Label(self.f, text="fraud", bg="goldenrod", width=30, height=2)
        self.fraud.pack(padx=20, pady=8)

        self.submit = Button(self.f, text="submit", height=2, width=30, bg="goldenrod", command=self.getvaluee)
        self.submit.pack(padx=20, pady=10)

        self.ff = Frame(newWindow, height="900", width="300", bg='blue')
        self.ff.pack(side=TOP)

        self.trustlevell = StringVar(value="1")
        self.totalScanTimeInSecondss = StringVar(value="770")
        self.grandTotall = StringVar(value="11.09")
        self.lineItemVoidss = StringVar(value="11")
        self.scansWithoutRegistrationn = StringVar(value="5")
        self.scannedLineItemsPerSecondd = StringVar(value="2")
        self.quantityModificationn = StringVar(value="0.033")
        self.valuePerSecondd = StringVar(value="0.01442")
        self.lineItemVoidsPerPositionn = StringVar(value="0.42")
        self.fraudd = StringVar()

        self.fff = Frame(self.ff, height="900", width="300")
        self.fff.pack(side=LEFT)
        self.trustlevel = Entry(self.fff, width=35, textvariable=self.trustlevell)
        self.trustlevel.pack(padx=20, pady=20)

        self.totalScanTimeInSecondsss = Entry(self.fff, width=35, textvariable=self.totalScanTimeInSecondss)
        self.totalScanTimeInSecondsss.pack(pady=15, padx=20)

        self.grandTotalll = Entry(self.fff, textvariable=self.grandTotall, width=35)
        self.grandTotalll.pack(pady=10, padx=20)

        self.lineItemVoidsss = Entry(self.fff, textvariable=self.lineItemVoidss, width=35)
        self.lineItemVoidsss.pack(pady=17, padx=20)

        self.scansWithoutRegistrationnn = Entry(self.fff, textvariable=self.scansWithoutRegistrationn, width=35)
        self.scansWithoutRegistrationnn.pack(pady=17, padx=20)

        self.quantityModificationnn = Entry(self.fff, textvariable=self.quantityModificationn, width=35)
        self.quantityModificationnn.pack(pady=17, padx=20)

        self.scannedLineItemsPerSeconddd = Entry(self.fff, textvariable=self.scannedLineItemsPerSecondd, width=35)
        self.scannedLineItemsPerSeconddd.pack(pady=17, padx=20)

        self.valuePerSeconddd = Entry(self.fff, text=self.valuePerSecondd, width=35)
        self.valuePerSeconddd.pack(padx=20, pady=17)

        self.lineItemVoidsPerPositionnn = Entry(self.fff, textvariable=self.lineItemVoidsPerPositionn, width=35)
        self.lineItemVoidsPerPositionnn.pack(padx=20, pady=17)

        self.frauddd = Entry(self.fff, textvariable=self.fraudd, width=35)
        self.frauddd.pack(padx=20, pady=17)
        newWindow.mainloop()

    def getvaluee(self):
        trustlevellll = self.trustlevell.get()
        totalScanTimeInSecondssss = self.totalScanTimeInSecondss.get()
        grandTotallll = self.grandTotall.get()
        lineItemVoidssss = self.lineItemVoidss.get()
        scansWithoutRegistrationnnn = self.scansWithoutRegistrationn.get()
        scannedLineItemsPerSecondddd = self.scannedLineItemsPerSecondd.get()
        quantityModificationnnn = self.quantityModificationn.get()
        valuePerSecondddd = self.valuePerSecondd.get()
        lineItemVoidsPerPositionnnn = self.lineItemVoidsPerPositionn.get()

        self.ada = AdaBoostClassifier(n_estimators=100)
        self.ada.fit(self.X_train, self.Y_train)
        self.predictions = self.ada.predict(self.X_validation)
        self.Y_pred = self.ada.predict([[float(trustlevellll), float(totalScanTimeInSecondssss), float(grandTotallll),
                                         float(lineItemVoidssss), float(scansWithoutRegistrationnnn),
                                         float(scannedLineItemsPerSecondddd)
                                            , float(quantityModificationnnn), float(valuePerSecondddd),
                                         float(lineItemVoidsPerPositionnnn)]])
        self.fraudd.set(self.Y_pred[len(self.Y_pred) - 1])




root = Tk()
root.title("FraudDetectionSystem")
f = FraudDetection(root)
root.mainloop()
